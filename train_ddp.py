import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from core.igev_stereo_encoder import IGEVStereo_encoder
import core.stereo_datasets as datasets
import core.fake_datasets as datasets_fake
import torch.nn.functional as F

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def setup_distributed(local_rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

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
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    mask0 = ((valid >= 0.5) & (mag < max_disp0)).unsqueeze(1)
    mask1 = ((valid >= 0.5) & (mag < max_disp1)).unsqueeze(1)
    mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert mask.shape == disp_gt.shape, [mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[mask.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(agg_preds[0][mask0.bool()], disp_gt[mask0.bool()], reduction='mean')
    disp_loss += 0.5 * F.smooth_l1_loss(agg_preds[1][mask1.bool()], disp_gt[mask1.bool()], reduction='mean')
    disp_loss += 0.2 * F.smooth_l1_loss(agg_preds[2][mask.bool()], disp_gt[mask.bool()], reduction='mean')

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[mask.bool()].mean()

    epe = torch.sum((iter_preds[-1] - disp_gt)**2, dim=1).sqrt()
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
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
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
        if rank == 0:
            self.writer = SummaryWriter(log_dir=self.logdir)

    def _print_training_status(self):
        if self.rank != 0:
            return
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.rank != 0:
            return
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.rank == 0:
            self.writer.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameters_size_in_MB(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


def unwrap_model(model):
    """Return the real model when using DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model


def clean_state_dict_keys(state_dict):
    """Remove common wrappers introduced by DDP or checkpoint wrappers."""
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
    """
    Save the full model state_dict.

    这个版本不再过滤 DINOv2。只要 Encoder 里把 DINOv2 vit_base 注册为 nn.Module
    或 nn.ModuleList，它就会自然进入 state_dict 并被保存到统一 ckpt。
    """
    raw_state = unwrap_model(model).state_dict()
    return {key: value.detach().cpu() for key, value in raw_state.items()}


def resize_perception_frames_if_needed(state_dict, model):
    """
    Resize encoder.perception_frames from a checkpoint if its stored H/W differs from
    the current model initialization size.
    """
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
    """
    Save a unified training checkpoint:
    - model: full model weights, including DINOv2 vit_base if it is registered in Encoder
    - optimizer/scheduler/scaler: for resume training
    - step/args: metadata
    """
    model_state = get_full_model_state_dict(model)

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "step": step,
        "args": vars(args),
        "contains_dinov2_vitb": any("dvt_vitb14" in k or "dinov2" in k.lower() for k in model_state.keys()),
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

    logging.info(f"Saving checkpoint to {save_path.absolute()}")
    logging.info(f"Checkpoint contains DINOv2 vit_base: {checkpoint['contains_dinov2_vitb']}")


def load_checkpoint_for_train(ckpt_path, model, optimizer=None, scheduler=None, scaler=None,
                              local_rank=0, resume_optimizer=False):
    """
    Load checkpoint saved by save_checkpoint.

    兼容两类 ckpt：
    1) 新的统一 ckpt: {"model": state_dict, ...}，包含 DINOv2 vit_base；
    2) 老的纯 state_dict 或不含 DINOv2 的 ckpt。
    """
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{local_rank}")

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

    if local_rank == 0:
        logging.info(f"Loaded checkpoint from {ckpt_path}")
        logging.info(f"load_state_dict message: {msg}")
        if isinstance(checkpoint, dict):
            if checkpoint.get("contains_dinov2_vitb") is not None:
                logging.info(f"Checkpoint contains DINOv2 vit_base: {checkpoint['contains_dinov2_vitb']}")
            if checkpoint.get("skipped_dinov2_keys") is not None:
                logging.info(f"Old checkpoint was saved without DINOv2 keys: {len(checkpoint['skipped_dinov2_keys'])}")

    return start_step


def train(local_rank, world_size, args):
    setup_distributed(local_rank, world_size)

    torch.manual_seed(666 + local_rank)
    np.random.seed(666 + local_rank)

    if local_rank == 0:
        Path(args.logdir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        )

    # 1. Build raw model first.
    raw_model = IGEVStereo_encoder(args).to(local_rank)

    if local_rank == 0:
        print("Parameter Count: %d" % count_parameters(raw_model))
        for name, param in raw_model.named_parameters():
            print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")
        total_params = count_parameters(raw_model)
        trainable_params = count_trainable_parameters(raw_model)
        param_size_MB = parameters_size_in_MB(raw_model)
        trainable_ratio = trainable_params / total_params * 100
        print(f"✅ 模型总参数量: {total_params:,}")
        print(f"✅ 可训练参数量: {trainable_params:,}")
        print(f"✅ 可训练参数占比: {trainable_ratio:.2f}%")
        print(f"✅ 参数总大小（float32）：{param_size_MB:.2f} MB")

    # 2. Load model weights before DDP wrapping.
    #    这里 strict=False，用于兼容旧 ckpt 或部分 key 不匹配的情况。
    start_step = 0
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
        if local_rank == 0:
            logging.info("Loading checkpoint weights before DDP wrapping...")
        start_step = load_checkpoint_for_train(
            args.restore_ckpt,
            raw_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            local_rank=local_rank,
            resume_optimizer=False,
        )

    # 3. Wrap with DDP.
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_dataset = datasets_fake.fetch_dataloader(args).dataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 4. Optimizer and scheduler.
    #    原代码里的 args.whichmodel 没有定义，而且 else 分支 optimizer 未创建，会直接报错。
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)

    # Optional: resume optimizer/scheduler/scaler states after they are created.
    if args.restore_ckpt is not None and args.resume_optimizer:
        _ = load_checkpoint_for_train(
            args.restore_ckpt,
            unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            local_rank=local_rank,
            resume_optimizer=True,
        )

    logger = Logger(model, scheduler, args.logdir, local_rank)
    logger.total_steps = start_step

    model.train()
    model.module.freeze_bn()

    total_steps = start_step
    global_batch_num = start_step

    while total_steps < args.num_steps:
        train_sampler.set_epoch(total_steps)

        for _, *data_blob in enumerate(tqdm(train_loader, disable=local_rank != 0)):
            if total_steps >= args.num_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            image1, image2, disp_gt, valid = [x.to(local_rank, non_blocking=True) for x in data_blob]
            agg_preds, iter_preds = model(image1, image2, iters=args.train_iters)
            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid)

            if local_rank == 0:
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

            # Save full unified checkpoint, including DINOv2 vit_base.
            if total_steps % args.save_freq == 0 and local_rank == 0:
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

    if local_rank == 0:
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
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Aquastereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training")
    parser.add_argument('--train_datasets', default='mix_raw_vidtome', help="training datasets")
    parser.add_argument('--lr', type=float, default=0.0001, help="max learning rate")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule")
    parser.add_argument('--image_size', type=int, nargs='+', default=[736,736], help="size of the random image crops")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel threads')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation')
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
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
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    parser.add_argument('--num_perception_frame', type=int, default=2, help='Number of perception frames')
    parser.add_argument('--pretrained', default=r'C:\Users\MECHREVO\Desktop\cvpr\opensource\AquaStereo-main\X3D_L.pyth', type=str, help='Path to pretrained weight')
    parser.add_argument('--use_dino', default=r"C:\Users\MECHREVO\Desktop\cvpr\opensource\AquaStereo-main\dinov2_vitb14_reg4_pretrain.pth")
    parser.add_argument('--save_freq', type=int, default=10000, help='checkpoint saving frequency')
    parser.add_argument('--resume_optimizer', action='store_true', help='resume optimizer/scheduler/scaler states from checkpoint')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs for DDP; default uses torch.cuda.device_count()')
    parser.add_argument('--vit_size', default='vitb', choices=['vits', 'vitb'], help='vit size')
    
    args = parser.parse_args()

    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    if args.world_size < 1:
        raise ValueError('No CUDA device is available for distributed training.')

    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    main()