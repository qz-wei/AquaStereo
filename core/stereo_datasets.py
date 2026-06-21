import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from PIL import Image
import numpy as np
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor,MY_FlowAugmentor
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader, DistributedSampler
class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.my_augmentor = MY_FlowAugmentor(**aug_params)
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024 

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:

            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sceneflow/', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'FlyingThings3D')
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/eth3d', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sintelstereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/tartanair'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        image1_list = sorted( glob(osp.join(root, '*/*/*/*/image_left/*.png')) )
        image2_list = sorted( glob(osp.join(root, '*/*/*/*/image_right/*.png')) )
        disp_list = sorted( glob(osp.join(root, '*/*/*/*/depth_left/*.npy')) )

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/instereo2k'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/kitti', image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        if year == 2012:
            root_12 = '/data/StereoDatasets/kitti/2012'
            image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ/000085_10.png')]*len(image1_list)

        if year == 2015:
            root_15 = '/data/StereoDatasets/kitti/2015'
            image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI_Rendered(StereoDataset):
    def __init__(self, aug_params=None,
                 root_img='/data/liangyingping_share/weiqizhe/xuanran_output',
                 root_depth='/data/liangyingping_share/weiqizhe/checkpoints/output_foundation_stereo',):

        super(KITTI_Rendered, self).__init__(aug_params, sparse=True, reader=None)
        self.resize_shape = (375, 1242)
        self.image_size=(375,1242)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size)  # 注意Resize接收的是 (H, W)
        self.image_list = []
        self.disparity_list = []

        dates = sorted(os.listdir(root_img))
        for date in dates:
            date_path = os.path.join(root_img, date)
            if not os.path.isdir(date_path):
                continue

            drives = sorted(os.listdir(date_path))
            for drive in drives:
                cam_left_path = os.path.join(date_path, drive, "image_02")
                cam_right_path = os.path.join(date_path, drive, "image_03")
                if not os.path.exists(cam_left_path) or not os.path.exists(cam_right_path):
                    continue

                frame_dirs = sorted(os.listdir(cam_left_path))
                for frame in frame_dirs:
                    frame_dir_left = os.path.join(cam_left_path, frame)
                    frame_dir_right = os.path.join(cam_right_path, frame)

                    render_imgs_left = sorted(glob(os.path.join(frame_dir_left, "*.png")))
                    render_imgs_right = sorted(glob(os.path.join(frame_dir_right, "*.png")))

                    if len(render_imgs_left) != len(render_imgs_right):
                        print(f"Warning: unmatched img count in {frame_dir_left} and {frame_dir_right}")
                        continue

                    for img_idx in range(min(3, len(render_imgs_left))):  # 三张以内
                        img_path_left = render_imgs_left[img_idx]
                        img_path_right = render_imgs_right[img_idx]

                        npy_path = os.path.join(root_depth, date, drive, frame, "disp_meter.npy")
                        if not os.path.exists(npy_path):
                            continue

                        self.image_list.append([img_path_left, img_path_right])
                        self.disparity_list.append(npy_path)

        assert len(self.image_list) > 0, "No rendered KITTI image pairs found!"
        print(f"Loaded {len(self.image_list)} rendered KITTI image pairs.")
    
    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # === 1. 读取 disp 为 npy ===
        disp = np.load(self.disparity_list[index]).astype(np.float32)
        disp[np.isinf(disp)] = 0
        # === 2. optional resize ===
        if hasattr(self, 'resize_shape'):
            from PIL import Image
            disp_img = Image.fromarray(disp)
            disp = cv2.resize(disp, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 3. 构造 valid 掩码 ===
        valid = (disp > 0) & (disp < 1024) & (~np.isnan(disp))

        # === 4. 构造 flow，只使用水平视差 ===
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = self.resize(Image.fromarray(img1))
        img2 = self.resize(Image.fromarray(img2))
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/vkitti2'):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/middlebury', split='2014', resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]        
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [[str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip


    train_dataset = None
    # for dataset_name in args.train_datasets:
    if args.train_datasets == 'sceneflow':
        aug_params['spatial_scale'] = False
        new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    elif args.train_datasets == 'vkitti2':
        new_dataset = VKITTI2(aug_params)
        logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")
    elif args.train_datasets == 'kitti':
        kitti12 = KITTI(aug_params, year=2012)
        logging.info(f"Adding {len(kitti12)} samples from KITTI 2012")
        kitti15 = KITTI(aug_params, year=2015)
        logging.info(f"Adding {len(kitti15)} samples from KITTI 2015")
        new_dataset = kitti12 + kitti15
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    elif args.train_datasets == 'render_kitti':
        new_dataset=KITTI_Rendered(aug_params,)
        logging.info("add render")
    elif args.train_datasets == 'distill_raw':
        new_dataset=distill_raw(aug_params,)
        logging.info("add distill_raw")
    elif args.train_datasets == 'mix_raw_vidtome':
        new_dataset=mix_raw_vidtome(aug_params,)
        logging.info("add mix_raw_vidtome")
    elif args.train_datasets == 'UWStereo':
        new_dataset=UWStereo(aug_params)
        logging.info("add UWStereo")
    elif args.train_datasets == 'render_video':
        new_dataset=video_render(aug_params)
        logging.info("add UWStereo")
    elif args.train_datasets == 'kitti_raw':
        new_dataset=KITTI_Raw(aug_params,)
        logging.info("add kitti_raw")    
    elif args.train_datasets == 'eth3d_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        sintel = SintelStereo(aug_params)
        logging.info(f"Adding {len(sintel)} samples from Sintel Stereo")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = tartanair + sceneflow + sintel * 50 + eth3d * 1000 + instereo2k  * 100 + crestereo * 2
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'eth3d_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")            
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'middlebury_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")             
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
    elif args.train_datasets == 'middlebury_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")                 
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 200 + fallingthings * 10
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")

    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    print(len(train_dataset))
    if args.distributed:
        sampler = DistributedSampler(train_dataset)
        shuffle = False  # 使用DistributedSampler时，不能再shuffle了
    else:
        sampler = None
        shuffle = True
    train_loader = data.DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=8,
                              drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader


class UWStereo(StereoDataset):
    def __init__(self, aug_params=None,istraining=True):
            """
            初始化 CoralDataset 类，读取图像和视差图的路径。
            
            Args:
                txt_file (str): 含有图像和视差图路径的文本文件。
                root_dir (str): 数据集的根目录，用于构造完整路径。
                transform (callable, optional): 用于数据增强的变换函数。
            """
            super(UWStereo, self).__init__(aug_params, sparse=True,)
            root_dir="/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/UWScene"
            if istraining==True:
                with open('/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/all_train.txt', 'r') as f:
                    lines = f.readlines()
            else:
                with open('/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/all_test.txt', 'r') as f:
                    lines = f.readlines()
            self.image_list = []
            self.disparity_list = []
            for line in lines:
                left_img, right_img, disp_img = line.strip().split()
                self.image_list.append([os.path.join(root_dir, left_img), os.path.join(root_dir, right_img)])
                self.disparity_list.append(os.path.join(root_dir, disp_img))

class validate_UWStereo(StereoDataset):
    def __init__(self, aug_params=None,istraining=False):
            """
            初始化 CoralDataset 类，读取图像和视差图的路径。
            
            Args:
                txt_file (str): 含有图像和视差图路径的文本文件。
                root_dir (str): 数据集的根目录，用于构造完整路径。
                transform (callable, optional): 用于数据增强的变换函数。
            """
            super(validate_UWStereo, self).__init__(aug_params, sparse=True,)
            root_dir="/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/UWScene"
            if istraining==True:
                with open('/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/all_train.txt', 'r') as f:
                    lines = f.readlines()
            else:
                with open('/data/liangyingping_share/weiqizhe/UWstereo/uwstereo/all_test.txt', 'r') as f:
                    lines = f.readlines()
                self.image_list = []
                self.disparity_list = []

                for line in lines:
                    left_img, right_img, disp_img = line.strip().split()
                    self.image_list.append([os.path.join(root_dir, left_img), os.path.join(root_dir, right_img)])
                    self.disparity_list.append(os.path.join(root_dir, disp_img))

class KITTI_Raw(StereoDataset):
    def __init__(self, aug_params=None,
                 root_img='/data/liangyingping_share/weiqizhe/kitty_raw_get/kitti_sync_data/kitti_sync_data',
                 root_depth='/data/liangyingping_share/weiqizhe/output_checkpoints/output_foundation_stereo',):
        super(KITTI_Raw, self).__init__(aug_params, sparse=True, reader=None)
        self.resize_shape = (375, 1242)
        self.image_size=(375,1242)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size)  # 注意Resize接收的是 (H, W)
        self.image_list = []
        self.disparity_list = []

        dates = sorted(os.listdir(root_img))
        for date in dates:
            date_path = os.path.join(root_img, date)
            if not os.path.isdir(date_path):
                continue

            drives = sorted(os.listdir(date_path))
            for drive in drives:
                cam_left_path = os.path.join(date_path, drive, "image_02")
                cam_right_path = os.path.join(date_path, drive, "image_03")
                if not os.path.exists(cam_left_path) or not os.path.exists(cam_right_path):
                    continue
                raw_dir_left=os.path.join(cam_left_path,'data')
                raw_dir_right=os.path.join(cam_right_path,'data')

                raw_imgs_left = sorted(glob(os.path.join(raw_dir_left, "*.png")))
                raw_imgs_right = sorted(glob(os.path.join(raw_dir_right, "*.png")))

                if len(raw_imgs_left) != len(raw_imgs_right):
                    print(f"Warning: unmatched img count in {raw_dir_left} and {raw_dir_right}")
                    continue

                for img_idx in range( len(raw_imgs_left)):  # 三张以内
                    img_path_left = raw_imgs_left[img_idx]
                    img_path_right = raw_imgs_right[img_idx]

                    npy_path = os.path.join(root_depth, date, drive, f"{img_idx:010d}", "disp_meter.npy")
                    if not os.path.exists(npy_path):
                        continue

                    self.image_list.append([img_path_left, img_path_right])
                    self.disparity_list.append(npy_path)

        assert len(self.image_list) > 0, "No rendered KITTI image pairs found!"
        print(f"Loaded {len(self.image_list)} rendered KITTI image pairs.")
    
    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # === 1. 读取 disp 为 npy ===
        disp = np.load(self.disparity_list[index]).astype(np.float32)
        disp[np.isinf(disp)] = 0
        # === 2. optional resize ===
        if hasattr(self, 'resize_shape'):
            from PIL import Image
            disp_img = Image.fromarray(disp)
            disp = cv2.resize(disp, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 3. 构造 valid 掩码 ===
        valid = (disp > 0) & (disp < 1024) & (~np.isnan(disp))

        # === 4. 构造 flow，只使用水平视差 ===
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = self.resize(Image.fromarray(img1))
        img2 = self.resize(Image.fromarray(img2))
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class video_render(StereoDataset):
    def __init__(self, aug_params=None,
                 root_img='/data/liangyingping_share/weiqizhe/output_render_videos',
                 root_depth='/data/liangyingping_share/weiqizhe/output_checkpoints/output_foundation_stereo'):

        super(video_render, self).__init__(aug_params, sparse=True, reader=None)
        
        self.image_size = (375, 1242) # 使用一个变量即可
        self.resize_shape = (375, 1242)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size)
        
        self.image_list = []      # 存储 [左图路径, 右图路径]
        self.disparity_list = []  # 存储 .npy 文件的路径

        print(f"Loading data from: {root_img} and {root_depth}")

        dates = sorted(os.listdir(root_img))
        for date in dates:
            print(f"Processing date: {date}")
            date_img_path = os.path.join(root_img, date)
            date_depth_path = os.path.join(root_depth, date)

            if not os.path.isdir(date_img_path):
                continue
            for drive in sorted(os.listdir(date_img_path)):
                drive_img_path=os.path.join(date_img_path,drive)
                drive_depth_path = os.path.join(date_depth_path, drive)
                # 遍历子段路径，如 2011_09_26_drive_0001_pairs_0000_0004
                sub_segments = sorted(os.listdir(drive_img_path))
                for seg in sub_segments:
                    seg_path = os.path.join(drive_img_path, seg)
                    frames_path = os.path.join(seg_path, "output", "vector", "frames")
                    if not os.path.exists(frames_path):
                        print(f"  Skipping segment {seg}: frames directory not found.")
                        continue

                    image_files = sorted(f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg')))

                    for i in range(0, len(image_files)-1, 2):  # 一对一对取
                        left_img_name = image_files[i]
                        right_img_name = image_files[i+1]

                        img_path_left = os.path.join(frames_path, left_img_name)
                        img_path_right = os.path.join(frames_path, right_img_name)

                        if not os.path.exists(img_path_right):
                            print(f"  Warning: Missing right image for {img_path_left}. Skipping pair.")
                            continue
                        #print(img_path_left)
                        # 提取帧号，构造 disp 路径
                        target_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path_left)))))
                        # 分割，提取起始帧
                        parts = target_dir.split('_')  # ['2011', '09', '26', 'drive', '0014', 'pairs', '0020', '0024']
                        start_frame = parts[-2]        # '0020'
                        npy_base_name=str(int(start_frame)+i//2).zfill(10)
                        #npy_base_name = os.path.splitext(left_img_name)[0].zfill(10)  # e.g., 0000

                        # 从 seg 中提取出原始 drive 名
                        # e.g., '2011_09_26_drive_0001_pairs_0000_0004' → '2011_09_26_drive_0001'
                        drive_base = seg.split('_pairs')[0]
                        depth_drive_name = drive_base + "_sync"

                        npy_path = os.path.join(date_depth_path, depth_drive_name, npy_base_name, "disp_meter.npy")

                        if not os.path.exists(npy_path):
                            print(f"  Warning: Missing disparity file {npy_path}. Skipping pair.")
                            continue

                        self.image_list.append([img_path_left, img_path_right])
                        self.disparity_list.append(npy_path)
                        #print(img_path_left,img_path_right)
                        #print(npy_path)
                        #print("*"*150)

            assert len(self.image_list) > 0, "No rendered KITTI image pairs found after checking all paths!"
            print(f"\nSuccessfully loaded {len(self.image_list)} rendered KITTI image and disparity pairs.")


    
    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # === 1. 读取 disp 为 npy ===
        disp = np.load(self.disparity_list[index]).astype(np.float32)
        disp[np.isinf(disp)] = 0
        # === 2. optional resize ===
        if hasattr(self, 'resize_shape'):
            from PIL import Image
            disp_img = Image.fromarray(disp)
            disp = cv2.resize(disp, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 3. 构造 valid 掩码 ===
        valid = (disp > 0) & (disp < 1024) & (~np.isnan(disp))

        # === 4. 构造 flow，只使用水平视差 ===
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = self.resize(Image.fromarray(img1))
        img2 = self.resize(Image.fromarray(img2))
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class mix_raw_vidtome(StereoDataset):
    def __init__(self, aug_params=None):
        root_img='/data/liangyingping_share/weiqizhe/kitty_raw_get/kitti_sync_data/kitti_sync_data'
        root_depth='/data/liangyingping_share/weiqizhe/output_checkpoints/output_foundation_stereo'
        super(mix_raw_vidtome, self).__init__(aug_params, sparse=True, reader=None)
        self.resize_shape = (384, 1216)
        self.image_size=(384,1216)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size)  # 注意Resize接收的是 (H, W)
        self.image_list = []
        self.disparity_list = []

        dates = sorted(os.listdir(root_img))
        for date in dates:
            date_path = os.path.join(root_img, date)
            if not os.path.isdir(date_path):
                continue

            drives = sorted(os.listdir(date_path))
            for drive in drives:
                cam_left_path = os.path.join(date_path, drive, "image_02")
                cam_right_path = os.path.join(date_path, drive, "image_03")
                if not os.path.exists(cam_left_path) or not os.path.exists(cam_right_path):
                    continue
                raw_dir_left=os.path.join(cam_left_path,'data')
                raw_dir_right=os.path.join(cam_right_path,'data')

                raw_imgs_left = sorted(glob(os.path.join(raw_dir_left, "*.png")))
                raw_imgs_right = sorted(glob(os.path.join(raw_dir_right, "*.png")))

                if len(raw_imgs_left) != len(raw_imgs_right):
                    print(f"Warning: unmatched img count in {raw_dir_left} and {raw_dir_right}")
                    continue

                for img_idx in range( len(raw_imgs_left)):  # 三张以内
                    img_path_left = raw_imgs_left[img_idx]
                    img_path_right = raw_imgs_right[img_idx]

                    npy_path = os.path.join(root_depth, date, drive, f"{img_idx:010d}", "disp_meter.npy")
                    if not os.path.exists(npy_path):
                        continue

                    self.image_list.append([img_path_left, img_path_right])
                    self.disparity_list.append(npy_path)

        assert len(self.image_list) > 0, "No rendered KITTI image pairs found!"
        print(f"Loaded {len(self.image_list)} rendered KITTI image pairs.")

        root_img='/data/liangyingping_share/weiqizhe/output_render_videos'
        root_depth='/data/liangyingping_share/weiqizhe/output_checkpoints/output_foundation_stereo'

        print(f"Loading data from: {root_img} and {root_depth}")

        dates = sorted(os.listdir(root_img))
        for date in dates:
            print(f"Processing date: {date}")
            date_img_path = os.path.join(root_img, date)
            date_depth_path = os.path.join(root_depth, date)

            if not os.path.isdir(date_img_path):
                continue
            for drive in sorted(os.listdir(date_img_path)):
                drive_img_path=os.path.join(date_img_path,drive)
                drive_depth_path = os.path.join(date_depth_path, drive)
                # 遍历子段路径，如 2011_09_26_drive_0001_pairs_0000_0004
                sub_segments = sorted(os.listdir(drive_img_path))
                for seg in sub_segments:
                    seg_path = os.path.join(drive_img_path, seg)
                    frames_path = os.path.join(seg_path, "output", "vector", "frames")
                    if not os.path.exists(frames_path):
                        print(f"  Skipping segment {seg}: frames directory not found.")
                        continue

                    image_files = sorted(f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg')))

                    for i in range(0, len(image_files)-1, 2):  # 一对一对取
                        left_img_name = image_files[i]
                        right_img_name = image_files[i+1]

                        img_path_left = os.path.join(frames_path, left_img_name)
                        img_path_right = os.path.join(frames_path, right_img_name)

                        if not os.path.exists(img_path_right):
                            print(f"  Warning: Missing right image for {img_path_left}. Skipping pair.")
                            continue
                        #print(img_path_left)
                        # 提取帧号，构造 disp 路径
                        target_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path_left)))))
                        # 分割，提取起始帧
                        parts = target_dir.split('_')  # ['2011', '09', '26', 'drive', '0014', 'pairs', '0020', '0024']
                        start_frame = parts[-2]        # '0020'
                        npy_base_name=str(int(start_frame)+i//2).zfill(10)
                        #npy_base_name = os.path.splitext(left_img_name)[0].zfill(10)  # e.g., 0000

                        # 从 seg 中提取出原始 drive 名
                        # e.g., '2011_09_26_drive_0001_pairs_0000_0004' → '2011_09_26_drive_0001'
                        drive_base = seg.split('_pairs')[0]
                        depth_drive_name = drive_base + "_sync"

                        npy_path = os.path.join(date_depth_path, depth_drive_name, npy_base_name, "disp_meter.npy")

                        if not os.path.exists(npy_path):
                            print(f"  Warning: Missing disparity file {npy_path}. Skipping pair.")
                            continue

                        self.image_list.append([img_path_left, img_path_right])
                        self.disparity_list.append(npy_path)
                        #print(img_path_left,img_path_right)
                        #print(npy_path)
                        #print("*"*150)

            assert len(self.image_list) > 0, "No rendered KITTI image pairs found after checking all paths!"
            print(f"\nSuccessfully loaded {len(self.image_list)} rendered KITTI image and disparity pairs.")
    
    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # === 1. 读取 disp 为 npy ===
        disp = np.load(self.disparity_list[index]).astype(np.float32)
        disp[np.isinf(disp)] = 0
        # === 2. optional resize ===
        if hasattr(self, 'resize_shape'):
            from PIL import Image
            disp_img = Image.fromarray(disp)
            disp = cv2.resize(disp, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 3. 构造 valid 掩码 ===
        valid = (disp > 0) & (disp < 1024) & (~np.isnan(disp))

        # === 4. 构造 flow，只使用水平视差 ===
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = self.resize(Image.fromarray(img1))
        img2 = self.resize(Image.fromarray(img2))
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)



class distill_raw(StereoDataset):
    def __init__(self, aug_params=None,
                 root_img='/data/liangyingping_share/weiqizhe/kitty_raw_get/kitti_sync_data/kitti_sync_data',
                 root_depth='/data/liangyingping_share/weiqizhe/output_checkpoints/output_foundation_stereo',):
        super(distill_raw, self).__init__(aug_params, sparse=True, reader=None)
        self.resize_shape = (375, 1242)
        self.image_size=(375,1242)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size)  # 注意Resize接收的是 (H, W)
        self.image_list = []
        self.disparity_list = []
        self.student_paths = []
        with open("/data/liangyingping_share/weiqizhe/train_dvt/kitti_rendere.txt", 'r') as f:
            for line in f:
                t_path, s_path = line.strip().split()
                self.student_paths.append([t_path, s_path])
        dates = sorted(os.listdir(root_img))
        for date in dates:
            date_path = os.path.join(root_img, date)
            if not os.path.isdir(date_path):
                continue

            drives = sorted(os.listdir(date_path))
            for drive in drives:
                cam_left_path = os.path.join(date_path, drive, "image_02")
                cam_right_path = os.path.join(date_path, drive, "image_03")
                if not os.path.exists(cam_left_path) or not os.path.exists(cam_right_path):
                    continue
                raw_dir_left=os.path.join(cam_left_path,'data')
                raw_dir_right=os.path.join(cam_right_path,'data')

                raw_imgs_left = sorted(glob(os.path.join(raw_dir_left, "*.png")))
                raw_imgs_right = sorted(glob(os.path.join(raw_dir_right, "*.png")))

                if len(raw_imgs_left) != len(raw_imgs_right):
                    print(f"Warning: unmatched img count in {raw_dir_left} and {raw_dir_right}")
                    continue

                for img_idx in range( len(raw_imgs_left)):  # 三张以内
                    img_path_left = raw_imgs_left[img_idx]
                    img_path_right = raw_imgs_right[img_idx]

                    npy_path = os.path.join(root_depth, date, drive, f"{img_idx:010d}", "disp_meter.npy")
                    if not os.path.exists(npy_path):
                        continue

                    self.image_list.append([img_path_left, img_path_right])
                    self.disparity_list.append(npy_path)

        assert len(self.image_list) > 0, "No rendered KITTI image pairs found!"
        print(f"Loaded {len(self.image_list)} rendered KITTI image pairs.")
    

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # === 1. 读取 disp 为 npy ===
        disp = np.load(self.disparity_list[index]).astype(np.float32)
        disp[np.isinf(disp)] = 0
        # === 2. optional resize ===
        if hasattr(self, 'resize_shape'):
            from PIL import Image
            disp_img = Image.fromarray(disp)
            disp = cv2.resize(disp, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 3. 构造 valid 掩码 ===
        valid = (disp > 0) & (disp < 1024) & (~np.isnan(disp))

        # === 4. 构造 flow，只使用水平视差 ===
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)


        img3 = frame_utils.read_gen(self.student_paths[index][0])
        img4 = frame_utils.read_gen(self.student_paths[index][1])

        img3 = np.array(img3).astype(np.uint8)
        img4 = np.array(img4).astype(np.uint8)
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = self.resize(Image.fromarray(img1))
        img2 = self.resize(Image.fromarray(img2))
        img1 = np.array(img1)
        img2 = np.array(img2)
        

        if len(img3.shape) == 2:
            img3 = np.tile(img3[..., None], (1, 1, 3))
            img4 = np.tile(img4[..., None], (1, 1, 3))
        else:
            img3 = img3[..., :3]
            img4 = img4[..., :3]

        img3 = self.resize(Image.fromarray(img3))
        img4 = self.resize(Image.fromarray(img4))
        img3 = np.array(img3)
        img4 = np.array(img4)
        

        img1,img2,img3,img4,flow,valid=self.my_augmentor(img1,img2,img3,img4,flow,valid)
        #print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}, img3 shape: {img3.shape},img4 shape: {img4.shape}disp_gt shape: {flow.shape}, valid shape: {valid.shape}")
            
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]

        #img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
        #img4 = torch.from_numpy(img4).permute(2, 0, 1).float()

        if self.img_pad is not None:
            img3 = F.pad(img3, [padW]*2 + [padH]*2)
            img4 = F.pad(img4, [padW]*2 + [padH]*2)

        return self.image_list[index] + [self.disparity_list[index]], img1, img2,img3,img4,flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)
