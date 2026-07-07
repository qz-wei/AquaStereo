import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from core.AquaStereo import AquaStereo
import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
        def state_dict(self):
            return None
        def load_state_dict(self, state_dict):
            pass


def sequence_loss(args, agg_preds, iter_preds, disp_gt, valid, loss_gamma=0.9):
    n_predictions = len(iter_preds)
    if ('kitti' in args.train_datasets) or ('eth3d' in args.train_datasets) or ('render_kitti' in args.train_datasets):
        max_disp0 = 192
        max_disp1 = 192
        max_disp = 192
    else:
        max_disp0 = 192
        max_disp1 = 384
        max_disp = 700

    disp_loss = 0.0
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    mask0 = ((valid >= 0.5) & (mag < max_disp0)).unsqueeze(1)
    mask1 = ((valid >= 0.5) & (mag < max_disp1)).unsqueeze(1)
    mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)

    assert mask.shape == disp_gt.shape, [mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[mask.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(agg_preds[0][mask0.bool()], disp_gt[mask0.bool()], reduction='mean')
    disp_loss += 0.5 * F.smooth_l1_loss(agg_preds[1][mask1.bool()], disp_gt[mask1.bool()], reduction='mean')
    disp_loss += 0.2 * F.smooth_l1_loss(agg_preds[2][mask.bool()], disp_gt[mask.bool()], reduction='mean')

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[mask.bool()].mean()

    epe = torch.sum((iter_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[mask.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return disp_loss, metrics


def fetch_optimizer(args, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='linear',
    )
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=self.logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_size_in_MB(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


def unwrap_model(model):
    return model


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = value
    return cleaned


def get_full_model_state_dict(model):
    raw_state = unwrap_model(model).state_dict()
    return {key: value.detach().cpu() for key, value in raw_state.items()}


def resize_perception_frames_if_needed(state_dict, model):
    model_state = unwrap_model(model).state_dict()
    candidate_keys = [
        "encoder.perception_frames",
        "module.encoder.perception_frames",
        "perception_frames",
    ]

    for key in candidate_keys:
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key in state_dict and clean_key in model_state and state_dict[clean_key].shape != model_state[clean_key].shape:
            src = state_dict[clean_key]
            dst = model_state[clean_key]
            _, C, T, H0, W0 = src.shape
            target_h, target_w = dst.shape[-2:]
            src_2d = src.permute(0, 2, 1, 3, 4).reshape(-1, C, H0, W0).float()
            src_2d = F.interpolate(src_2d, size=(target_h, target_w), mode="bilinear", align_corners=False)
            src = src_2d.reshape(1, T, C, target_h, target_w).permute(0, 2, 1, 3, 4).to(dtype=dst.dtype)
            state_dict[clean_key] = src

    return state_dict


def save_checkpoint(model, optimizer, scheduler, scaler, args, step, save_path):
    model_state = get_full_model_state_dict(model)
    checkpoint = {
        "model_state": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "step": step,
        "args": vars(args),
        "vit_size": getattr(args, "vit_size", "vitb"),
        "contains_dinov2": any("dvt_vitb14" in k or "dinov2" in k.lower() for k in model_state.keys()),
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

    logging.info(f"Saving checkpoint to {save_path.absolute()}")
    logging.info(f"Checkpoint DINO backbone: {checkpoint['vit_size']}")
    logging.info(f"Checkpoint contains DINOv2: {checkpoint['contains_dinov2']}")


def load_checkpoint_for_train(ckpt_path, model, optimizer=None, scheduler=None, scaler=None,
                              device="cuda:0", resume_optimizer=False):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint

    model_state = clean_state_dict_keys(model_state)
    model_state = resize_perception_frames_if_needed(model_state, model)
    msg = model.load_state_dict(model_state, strict=False)

    start_step = 0
    if isinstance(checkpoint, dict):
        start_step = int(checkpoint.get("step", 0) or 0)

        if resume_optimizer:
            if optimizer is not None and checkpoint.get("optimizer") is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(checkpoint["scaler"])

    logging.info(f"Loaded checkpoint from {ckpt_path}")
    logging.info(f"load_state_dict message: {msg}")
    if isinstance(checkpoint, dict):
        if checkpoint.get("vit_size") is not None:
            logging.info(f"Checkpoint DINO backbone: {checkpoint['vit_size']}")
        elif isinstance(checkpoint.get("args"), dict) and checkpoint["args"].get("vit_size") is not None:
            logging.info(f"Checkpoint DINO backbone: {checkpoint['args']['vit_size']}")

        if checkpoint.get("contains_dinov2") is not None:
            logging.info(f"Checkpoint contains DINOv2: {checkpoint['contains_dinov2']}")
        elif checkpoint.get("contains_dinov2_vitb") is not None:
            logging.info(f"Checkpoint contains DINOv2 vit_base: {checkpoint['contains_dinov2_vitb']}")

    return start_step


def normalize_optional_path(path):
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in ["", "none", "null"]:
        return None
    return path


def read_checkpoint_args(ckpt_path):
    """Read metadata args from a checkpoint without constructing the model."""
    ckpt_path = normalize_optional_path(ckpt_path)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return {}
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        logging.warning(f"Failed to read checkpoint metadata from {ckpt_path}: {exc}")
        return {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def resolve_init_args(args):
    """
    Resolve initialization policy before AquaStereo(args) is constructed.

    Rules:
    - If restore_ckpt is provided, the full model should be initialized from that ckpt.
      External pretrained_change3d/pretrained_dino paths are ignored by Encoder.
    - If restore_ckpt is not provided, initialize X3D/DINO from external pretrained paths.
    - vit_size can be inferred from checkpoint metadata when restore_ckpt is provided.
    """
    args.restore_ckpt = normalize_optional_path(args.restore_ckpt)

    ckpt_args = read_checkpoint_args(args.restore_ckpt) if args.restore_ckpt is not None else {}

    if args.vit_size is None:
        ckpt_vit_size = ckpt_args.get("vit_size")
        if ckpt_vit_size in ["vits", "vitb"]:
            args.vit_size = ckpt_vit_size
        else:
            args.vit_size = "vitb"

    if args.vit_size not in ["vits", "vitb"]:
        raise ValueError(f"Unsupported --vit_size {args.vit_size!r}; expected 'vits' or 'vitb'.")

    args.pretrained_change3d = normalize_optional_path(args.pretrained_change3d)
    args.pretrained_dino = normalize_optional_path(args.pretrained_dino)

    # When training from external pretrain, auto-select DINO path by vit_size if --pretrained_dino is omitted.
    if args.restore_ckpt is None and args.pretrained_dino is None:
        if args.vit_size == "vits":
            args.pretrained_dino = getattr(args, "pretrained_dino_vits", "./pretrained/dinov2_vits14_pretrain.pth")
        else:
            args.pretrained_dino = getattr(args, "pretrained_dino_vitb", "./pretrained/dinov2_vitb14_pretrain.pth")

    return args


def parse_batch(data_blob, device):
    if len(data_blob) == 5:
        _, image1, image2, disp_gt, valid = data_blob
    elif len(data_blob) == 4:
        image1, image2, disp_gt, valid = data_blob
    else:
        raise ValueError(f"Unexpected batch format, len(data_blob)={len(data_blob)}")

    image1 = image1.to(device, non_blocking=True)
    image2 = image2.to(device, non_blocking=True)
    disp_gt = disp_gt.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    return image1, image2, disp_gt, valid


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("No CUDA device is available for single-GPU training.")

    torch.cuda.set_device(device)
    torch.manual_seed(666)
    np.random.seed(666)

    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    )

    args = resolve_init_args(args)
    restore_ckpt = args.restore_ckpt

    model = AquaStereo(args).to(device)

    start_step = 0
    logging.info(f"Using DINO backbone: {args.vit_size}")
    if restore_ckpt is not None:
        assert restore_ckpt.endswith(".pth") or restore_ckpt.endswith(".pt")
        logging.info(f"restore_ckpt is set; model weights will come from checkpoint: {restore_ckpt}")
        logging.info("External --pretrained_change3d and --pretrained_dino are ignored when restore_ckpt is set.")
    else:
        logging.info(f"No restore_ckpt; initialize X3D from: {args.pretrained_change3d}")
        logging.info(f"No restore_ckpt; initialize DINO from: {args.pretrained_dino}")

    train_loader = datasets.fetch_dataloader(args)

    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision)

    if restore_ckpt is not None:
        start_step = load_checkpoint_for_train(
            restore_ckpt,
            model,
            optimizer=optimizer if args.resume_optimizer else None,
            scheduler=scheduler if args.resume_optimizer else None,
            scaler=scaler if args.resume_optimizer else None,
            device=device,
            resume_optimizer=args.resume_optimizer,
        )
        if not args.resume_optimizer:
            # Use restore_ckpt as weight initialization for a new training run.
            # Add --resume_optimizer if you want true interrupted-training resume.
            start_step = 0

    logger = Logger(model, scheduler, args.logdir)
    logger.total_steps = start_step

    model.train()
    model.freeze_bn()

    total_steps = start_step
    global_batch_num = start_step

    while total_steps < args.num_steps:
        for data_blob in tqdm(train_loader):
            if total_steps >= args.num_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            image1, image2, disp_gt, valid = parse_batch(data_blob, device)
            agg_preds, iter_preds = model(image1, image2, iters=args.train_iters)
            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid)

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            total_steps += 1

            if total_steps % args.save_freq == 0:
                save_path = Path(args.logdir) / f"{total_steps}_{args.name}.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    step=total_steps,
                    save_path=save_path,
                )

    print("FINISHED TRAINING")
    final_path = Path(args.logdir) / f"{args.name}.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        step=total_steps,
        save_path=final_path,
    )

    logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Aquastereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training")
    parser.add_argument('--train_datasets', default='kitti', help="training datasets")
    parser.add_argument('--lr', type=float, default=0.0001, help="max learning rate")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320,736], help="size of the random image crops")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel threads')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation')
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range")
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8], help='re-scale the images')
    parser.add_argument('--noyjitter', action='store_true', help="don't simulate imperfect rectification")
    parser.add_argument('--num_perception_frame', type=int, default=2, help='Number of perception frames')
    parser.add_argument('--pretrained_change3d', default='./pretrained/X3D_L.pyth', type=str, help='Path to pretrained weight')
    parser.add_argument('--pretrained_dino', default='./pretrained', type=str, help='Path to DINOv2 pretrained weight or pretrained directory')
    parser.add_argument('--save_freq', type=int, default=10000, help='checkpoint saving frequency')
    parser.add_argument('--resume_optimizer', action='store_true', help='resume optimizer/scheduler/scaler states from checkpoint')
    parser.add_argument('--vit_size', default='vitb', choices=['vits', 'vitb'], help='DINO backbone size. If omitted with restore_ckpt, read from checkpoint args; otherwise default to vitb')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
