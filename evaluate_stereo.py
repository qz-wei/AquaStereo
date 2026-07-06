import sys
sys.path.append('core')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from core.AquaStereo import AquaStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))

        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = data.DataLoader(val_dataset, batch_size=8, 
        pin_memory=True, shuffle=False, num_workers=8)

    out_list, epe_list = [], []
    for i_batch, (_, *data_blob) in enumerate(tqdm(val_loader)):
        image1, image2, disp_gt, valid_gt = [x for x in data_blob]

        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu()
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.abs(disp_pr - disp_gt)

        epe = epe.flatten()
        val = (disp_gt.abs().flatten() < 768)
        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_sceneflow.txt', 'a')
    f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    print("Validation Scene Flow: %f, %f" % (epe, d1))
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='MiddEval3', resolution='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, resolution=resolution)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_middlebury.txt', 'a')
    f.write("Validation Middlebury: %f, %f\n" % (epe, d1))

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


def normalize_optional_path(path):
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in ["", "none", "null"]:
        return None
    return path


def read_checkpoint_metadata(ckpt_path):
    ckpt_path = normalize_optional_path(ckpt_path)
    if ckpt_path is None:
        return {}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"restore_ckpt not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        meta = {}
        if isinstance(checkpoint.get("args"), dict):
            meta.update(checkpoint["args"])
        if checkpoint.get("vit_size") is not None:
            meta["vit_size"] = checkpoint["vit_size"]
        return meta
    return {}


def resolve_eval_args(args):
    args.restore_ckpt = normalize_optional_path(args.restore_ckpt)
    if args.restore_ckpt is None:
        raise ValueError("Evaluation requires --restore_ckpt.")

    ckpt_args = read_checkpoint_metadata(args.restore_ckpt)
    if args.vit_size is None:
        ckpt_vit_size = ckpt_args.get("vit_size")
        if ckpt_vit_size in ["vits", "vitb"]:
            args.vit_size = ckpt_vit_size
        else:
            args.vit_size = "vitb"

    if args.vit_size not in ["vits", "vitb"]:
        raise ValueError(f"Unsupported --vit_size {args.vit_size!r}; expected 'vits' or 'vitb'.")

    return args


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def clean_key_for_dp(key):
    # Strip common wrappers first, then add one DataParallel prefix.
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return f"module.{key}"


def resize_perception_frames_if_needed(state_dict, model):
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key.endswith("encoder.perception_frames") and key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                src = state_dict[key]
                dst = model_state[key]
                _, C, T, H0, W0 = src.shape
                target_h, target_w = dst.shape[-2:]
                src_2d = src.permute(0, 2, 1, 3, 4).reshape(-1, C, H0, W0).float()
                src_2d = F.interpolate(src_2d, size=(target_h, target_w), mode="bilinear", align_corners=False)
                src = src_2d.reshape(1, T, C, target_h, target_w).permute(0, 2, 1, 3, 4).to(dtype=dst.dtype)
                state_dict[key] = src
    return state_dict


def load_checkpoint_for_eval(model, ckpt_path, strict=True):
    ckpt_path = normalize_optional_path(ckpt_path)
    if ckpt_path is None:
        raise ValueError("Evaluation requires --restore_ckpt.")
    if not (ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")):
        raise ValueError(f"restore_ckpt should be a .pth or .pt file, got: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"restore_ckpt not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_model_state(checkpoint)

    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    ignored = []
    for k, v in state_dict.items():
        name = clean_key_for_dp(k)
        if name in model_keys:
            new_state_dict[name] = v
        else:
            ignored.append(k)

    new_state_dict = resize_perception_frames_if_needed(new_state_dict, model)
    msg = model.load_state_dict(new_state_dict, strict=strict)
    logging.info(f"Loaded checkpoint from {ckpt_path}")
    logging.info(f"load_state_dict message: {msg}")
    if ignored:
        logging.info(f"Ignored {len(ignored)} checkpoint keys that are not used by the current model.")
    return msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--dataset', help="dataset for evaluation", default='kitti', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    parser.add_argument('--num_perception_frame', type=int, default=2, help='Number of perception frames')
    parser.add_argument('--vit_size', default='vits', choices=['vits', 'vitb'], help='DINO backbone size. If omitted, read from checkpoint args; otherwise default to vitb')
    args = parser.parse_args()
    args = resolve_eval_args(args)

    model = torch.nn.DataParallel(AquaStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    logging.info(f"Using DINO backbone: {args.vit_size}")
    logging.info("Loading checkpoint...")
    load_checkpoint_for_eval(model, args.restore_ckpt, strict=True)
    logging.info("Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)