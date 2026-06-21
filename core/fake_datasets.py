import argparse
import math
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler


class FakeStereoDataset(data.Dataset):
    """Standalone fake stereo dataset for quick smoke tests."""
    def __init__(self, length=64, image_size=(384, 1216), max_disp=192, variable_size=False, return_info=True, seed=666):
        self.length = int(length)
        self.base_h = int(image_size[0])
        self.base_w = int(image_size[1])
        self.max_disp = float(max_disp)
        self.variable_size = bool(variable_size)
        self.return_info = bool(return_info)
        self.seed = int(seed)
        self.image_list = [[f"fake_left_{i:06d}.png", f"fake_right_{i:06d}.png"] for i in range(self.length)]
        self.disparity_list = [f"fake_disp_{i:06d}.npy" for i in range(self.length)]

    def __len__(self):
        return self.length

    def _get_hw(self, index):
        if not self.variable_size:
            return self.base_h, self.base_w
        candidates = [
            (self.base_h, self.base_w),
            (max(64, self.base_h - 32), max(96, self.base_w - 64)),
            (self.base_h + 32, self.base_w + 64),
        ]
        return candidates[index % len(candidates)]

    def __getitem__(self, index):
        H, W = self._get_hw(index)
        g = torch.Generator().manual_seed(self.seed + int(index))
        img1 = torch.randint(0, 256, (3, H, W), generator=g, dtype=torch.int16).float()

        x = torch.linspace(0.0, 1.0, W).view(1, 1, W).expand(1, H, W)
        y = torch.linspace(0.0, 1.0, H).view(1, H, 1).expand(1, H, W)
        disp = 4.0 + 0.35 * self.max_disp * x + 2.0 * torch.sin(2.0 * math.pi * y)
        disp = disp + torch.randn((1, H, W), generator=g) * 0.25
        disp = disp.clamp(min=0.0, max=self.max_disp - 1.0).float()

        shift = max(1, int(min(self.max_disp // 16, W // 16)))
        img2 = torch.roll(img1, shifts=-shift, dims=2)
        img2 = (img2 + torch.randn((3, H, W), generator=g) * 2.0).clamp(0.0, 255.0).float()
        valid = torch.ones(H, W).float()

        if self.return_info:
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, disp, valid
        return img1, img2, disp, valid


def fetch_dataloader(args):
    dataset = FakeStereoDataset(
        length=8,
        image_size=[512, 512],
        max_disp=192,
        variable_size=False,
        return_info=True,
        seed=42,
    )
    sampler =None
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 1216])
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--fake_length', type=int, default=8)
    parser.add_argument('--fake_variable_size', action='store_true')
    parser.add_argument('--fake_return_info', action='store_true', default=True)
    parser.add_argument('--fake_seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    loader = fetch_fake_dataloader(args)
    batch = next(iter(loader))

    if args.fake_return_info:
        info, img1, img2, disp, valid = batch
        print('info example:', info)
    else:
        img1, img2, disp, valid = batch

    print('img1 :', tuple(img1.shape), img1.dtype, float(img1.min()), float(img1.max()))
    print('img2 :', tuple(img2.shape), img2.dtype, float(img2.min()), float(img2.max()))
    print('disp :', tuple(disp.shape), disp.dtype, float(disp.min()), float(disp.max()))
    print('valid:', tuple(valid.shape), valid.dtype, float(valid.min()), float(valid.max()))


if __name__ == '__main__':
    _demo()
