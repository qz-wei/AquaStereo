import os
import argparse
import logging
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
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


def setup_distributed(local_rank, world_size, master_addr="127.0.0.1", master_port="12355"):
    """Initialize single-node DDP."""
    os.environ.setdefault("MASTER_ADDR", str(master_addr))
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def setup_logger(logdir, rank):
    if not is_main_process(rank):
        return
    Path(logdir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )


def normalize_optional_path(path):
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in ["", "none", "null"]:
        return None
    return path


def read_checkpoint_args(ckpt_path):
    """Read checkpoint metadata before constructing the model."""
    ckpt_path = normalize_optional_path(ckpt_path)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return {}
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return {}
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("args"), dict):
            return checkpoint["args"]
        if checkpoint.get("vit_size") in ["vits", "vitb"]:
            return {"vit_size": checkpoint.get("vit_size")}
    return {}


def resolve_init_args(args):
    """
    Keep this consistent with the working single-GPU train.py.

    Rules:
    1. If restore_ckpt is provided, model weights come from that ckpt.
       Encoder will ignore external pretrained_change3d / pretrained_dino.
    2. If restore_ckpt is not provided, initialize X3D/DINO from external pretrain paths.
    3. If restore_ckpt contains args['vit_size'], use it when --vit_size is not explicitly set.
    """
    args.restore_ckpt = normalize_optional_path(args.restore_ckpt)
    args.pretrained_change3d = normalize_optional_path(args.pretrained_change3d)
    args.pretrained_dino = normalize_optional_path(args.pretrained_dino)

    ckpt_args = read_checkpoint_args(args.restore_ckpt) if args.restore_ckpt is not None else {}

    if args.vit_size is None:
        ckpt_vit_size = ckpt_args.get("vit_size")
        if ckpt_vit_size in ["vits", "vitb"]:
            args.vit_size = ckpt_vit_size
        else:
            args.vit_size = "vitb"

    if args.vit_size not in ["vits", "vitb"]:
        raise ValueError(f"Unsupported --vit_size {args.vit_size!r}; expected 'vits' or 'vitb'.")

    if args.restore_ckpt is None and args.pretrained_dino is None:
        args.pretrained_dino = "./pretrained"

    return args


def sequence_loss(args, agg_preds, iter_preds, disp_gt, valid, loss_gamma=0.9):
    n_predictions = len(iter_preds)

    if ("kitti" in args.train_datasets) or ("eth3d" in args.train_datasets) or ("render_kitti" in args.train_datasets):
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

    disp_loss += 1.0 * F.smooth_l1_loss(
        agg_preds[0][mask0.bool()], disp_gt[mask0.bool()], reduction="mean"
    )
    disp_loss += 0.5 * F.smooth_l1_loss(
        agg_preds[1][mask1.bool()], disp_gt[mask1.bool()], reduction="mean"
    )
    disp_loss += 0.2 * F.smooth_l1_loss(
        agg_preds[2][mask.bool()], disp_gt[mask.bool()], reduction="mean"
    )

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[mask.bool()].mean()

    epe = torch.sum((iter_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[mask.view(-1)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return disp_loss, metrics


def reduce_metrics(metrics, device):
    """Average scalar metrics across ranks for cleaner rank0 logging."""
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    reduced = {}
    world_size = dist.get_world_size()
    for key, value in metrics.items():
        t = torch.tensor(float(value), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[key] = (t / world_size).item()
    return reduced


def fetch_optimizer(args, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir, rank):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logdir = logdir
        self.rank = rank
        self.writer = SummaryWriter(log_dir=self.logdir) if is_main_process(rank) else None

    def _print_training_status(self):
        if not is_main_process(self.rank) or len(self.running_loss) == 0:
            return
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        if not is_main_process(self.rank):
            return
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_scalar(self, key, value, step):
        if is_main_process(self.rank) and self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def close(self):
        if is_main_process(self.rank) and self.writer is not None:
            self.writer.close()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


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


def get_full_model_state_dict(model):
    raw_state = unwrap_model(model).state_dict()
    return {key: value.detach().cpu() for key, value in raw_state.items()}


def checkpoint_contains_dino(model_state):
    return any(
        ("dvt_vitb14" in key) or ("dvt_vits14" in key) or ("dinov2" in key.lower())
        for key in model_state.keys()
    )


def save_checkpoint(model, optimizer, scheduler, scaler, args, step, save_path, rank):
    if not is_main_process(rank):
        return
    model_state = get_full_model_state_dict(model)
    checkpoint = {
        "model_state": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "step": step,
        "args": vars(args),
        "vit_size": getattr(args, "vit_size", "vitb"),
        "contains_dinov2": checkpoint_contains_dino(model_state),
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logging.info(f"Saving checkpoint to {save_path.absolute()}")
    logging.info(f"Checkpoint DINO backbone: {checkpoint['vit_size']}")
    logging.info(f"Checkpoint contains DINOv2: {checkpoint['contains_dinov2']}")


def load_checkpoint_for_train(
    ckpt_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    resume_optimizer=False,
    rank=0,
):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
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
    msg = unwrap_model(model).load_state_dict(model_state, strict=False)

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

    if is_main_process(rank):
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_loader(args, rank, world_size):
    base_loader = datasets.fetch_dataloader(args)
    train_dataset = base_loader.dataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, train_sampler


def train_worker(local_rank, world_size, args):
    setup_distributed(local_rank, world_size, args.master_addr, args.master_port)
    rank = local_rank
    device = torch.device(f"cuda:{local_rank}")

    setup_logger(args.logdir, rank)
    seed_everything(666 + rank)

    if is_main_process(rank):
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_main_process(rank):
        logging.info(f"World size: {world_size}")
        logging.info(f"Using DINO backbone: {args.vit_size}")
        if args.restore_ckpt is not None:
            logging.info(f"restore_ckpt is set; model weights will come from checkpoint: {args.restore_ckpt}")
            logging.info("External --pretrained_change3d and --pretrained_dino are ignored by Encoder when restore_ckpt is set.")
        else:
            logging.info(f"No restore_ckpt; initialize X3D from: {args.pretrained_change3d}")
            logging.info(f"No restore_ckpt; initialize DINO from: {args.pretrained_dino}")

    raw_model = AquaStereo(args).to(device)

    start_step = 0
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
        if is_main_process(rank):
            logging.info("Loading checkpoint weights before DDP wrapping...")
        start_step = load_checkpoint_for_train(
            args.restore_ckpt,
            raw_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location=device,
            resume_optimizer=False,
            rank=rank,
        )
        if not args.resume_optimizer:
            start_step = 0

    model = DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters,
    )

    train_loader, train_sampler = build_train_loader(args, rank, world_size)
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision)

    if args.restore_ckpt is not None and args.resume_optimizer:
        _ = load_checkpoint_for_train(
            args.restore_ckpt,
            unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
            resume_optimizer=True,
            rank=rank,
        )

    logger = Logger(model, scheduler, args.logdir, rank)
    logger.total_steps = start_step

    model.train()
    model.module.freeze_bn()

    total_steps = start_step
    global_batch_num = start_step
    epoch = 0

    while total_steps < args.num_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for data_blob in tqdm(train_loader, disable=not is_main_process(rank)):
            if total_steps >= args.num_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            image1, image2, disp_gt, valid = parse_batch(data_blob, device)
            agg_preds, iter_preds = model(image1, image2, iters=args.train_iters)
            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid)
            metrics = reduce_metrics(metrics, device)

            logger.write_scalar("live_loss", loss.item(), global_batch_num)
            logger.write_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_batch_num)
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
                save_checkpoint(model, optimizer, scheduler, scaler, args, total_steps, save_path, rank)

    if is_main_process(rank):
        print("FINISHED TRAINING")
        final_path = Path(args.logdir) / f"{args.name}.pth"
        save_checkpoint(model, optimizer, scheduler, scaler, args, total_steps, final_path, rank)

    logger.close()
    dist.barrier()
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Aquastereo", help="name your experiment")
    parser.add_argument("--restore_ckpt", default=None, help="load the weights from a specific checkpoint")
    parser.add_argument("--logdir", default="./checkpoints", help="the directory to save logs and checkpoints")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--precision_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Choose precision type")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    parser.add_argument("--train_datasets", default="kitti", help="training datasets")
    parser.add_argument("--lr", type=float, default=0.0001, help="max learning rate")
    parser.add_argument("--num_steps", type=int, default=200000, help="number of optimizer steps")
    parser.add_argument("--image_size", type=int, nargs="+", default=[320, 736], help="size of random image crops")
    parser.add_argument("--train_iters", type=int, default=22, help="number of updates to disparity field")
    parser.add_argument("--wdecay", type=float, default=0.00001, help="weight decay")
    parser.add_argument("--num_workers", type=int, default=1, help="dataloader workers per process")
    parser.add_argument("--valid_iters", type=int, default=32, help="validation forward iterations")

    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)

    parser.add_argument("--max_disp", type=int, default=768, help="max disparity range")
    parser.add_argument("--s_disp_range", type=int, default=48)
    parser.add_argument("--m_disp_range", type=int, default=96)
    parser.add_argument("--l_disp_range", type=int, default=192)
    parser.add_argument("--s_disp_interval", type=int, default=1)
    parser.add_argument("--m_disp_interval", type=int, default=2)
    parser.add_argument("--l_disp_interval", type=int, default=4)

    parser.add_argument("--img_gamma", type=float, nargs="+", default=None)
    parser.add_argument("--saturation_range", type=float, nargs="+", default=[0, 1.4])
    parser.add_argument("--do_flip", default=False, choices=["h", "v"])
    parser.add_argument("--spatial_scale", type=float, nargs="+", default=[-0.4, 0.8])
    parser.add_argument("--noyjitter", action="store_true")

    parser.add_argument("--num_perception_frame", type=int, default=2)
    parser.add_argument("--pretrained_change3d", default="./pretrained/X3D_L.pyth", type=str)
    parser.add_argument("--pretrained_dino", default="./pretrained", type=str)
    parser.add_argument("--vit_size", default=None, choices=["vits", "vitb"], help="DINO backbone size; if omitted with restore_ckpt, read checkpoint args, otherwise vitb")

    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--world_size", type=int, default=None, help="number of visible GPUs to use")
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", default="12355")
    parser.add_argument("--find_unused_parameters", action="store_true")

    args = parser.parse_args()
    args = resolve_init_args(args)

    if not torch.cuda.is_available():
        raise ValueError("No CUDA device is available for DDP training.")

    if args.world_size is None:
        args.world_size = torch.cuda.device_count()

    visible_gpu_count = torch.cuda.device_count()
    if args.world_size < 1:
        raise ValueError("world_size must be >= 1.")
    if args.world_size > visible_gpu_count:
        raise ValueError(
            f"--world_size={args.world_size}, but only {visible_gpu_count} CUDA devices are visible. "
            "Check CUDA_VISIBLE_DEVICES."
        )

    mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
