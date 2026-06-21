# 🚀 AquaStereo 🚀

<p align="center">
  <img src="fig/teaser.png" width="90%" alt="AquaStereo Teaser">
</p>

<div align="center">

**AquaStereo: [Paper Title or Short Description Here]**

<a href="YOUR_ARXIV_LINK_HERE">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv">
</a>
<a href="YOUR_HUGGINGFACE_LINK_HERE">
  <img src="https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface" alt="Hugging Face">
</a>
<a href="YOUR_DEMO_LINK_HERE">
  <img src="https://img.shields.io/badge/Demo-Video-blue" alt="Demo">
</a>

</div>

---

## 🌊 Overview

AquaStereo is a stereo matching framework designed for accurate and robust disparity estimation. It aims to improve stereo depth perception under challenging conditions and provide strong generalization ability across different scenes.

This repository contains the official implementation of **AquaStereo**, including model definition, training scripts, evaluation scripts, and visualization examples.

---

## 🤗 Demo

<p align="center">
  <a href="YOUR_DEMO_LINK_HERE">
    <img src="fig/teaser.png" width="70%" alt="AquaStereo Demo">
  </a>
</p>

Demo video link: **YOUR_DEMO_LINK_HERE**

---

## 🧠 Network Architecture

<p align="center">
  <img src="fig/network.png" width="95%" alt="AquaStereo Network">
</p>

The overall architecture of AquaStereo is shown above. The model takes a rectified stereo image pair as input and predicts the corresponding disparity map.

---

## 🌈 Visualization

<p align="center">
  <img src="fig/teaser.png" width="95%" alt="AquaStereo Visualization">
</p>

Qualitative visualization examples of AquaStereo.

---

## ⚙️ Installation

The environment follows the same setting as the reference implementation.

* NVIDIA RTX 3090
* Python 3.8
* CUDA 11.8

### Create a virtual environment

```bash
conda create -n aquastereo python=3.8
conda activate aquastereo
```

### Install dependencies

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib
pip install timm==0.6.13
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
```

---

## 📦 Model Weights

The pretrained weights will be released on Hugging Face.

|       Model      |                       Link                      |
| :--------------: | :---------------------------------------------: |
|    AquaStereo    | [Download 🤗](YOUR_HUGGINGFACE_MODEL_LINK_HERE) |
| AquaStereo-Large | [Download 🤗](YOUR_HUGGINGFACE_MODEL_LINK_HERE) |

Please place the downloaded weights under:

```bash
pretrained/
```

Example:

```text
pretrained/
└── aquastereo.pth
```

---

## ✏️ Required Data

The following datasets can be used for training and evaluation:

* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)
* [FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)

Please organize the datasets according to your local configuration and update the dataset paths in the corresponding config or script files.

---

## ✈️ Evaluation

To evaluate AquaStereo, run:

```bash
python evaluate_stereo.py --restore_ckpt ./pretrained/aquastereo.pth --dataset kitti
```

You can replace `kitti` with other supported datasets, for example:

```bash
python evaluate_stereo.py --restore_ckpt ./pretrained/aquastereo.pth --dataset sceneflow
python evaluate_stereo.py --restore_ckpt ./pretrained/aquastereo.pth --dataset eth3d
python evaluate_stereo.py --restore_ckpt ./pretrained/aquastereo.pth --dataset middlebury
```

---

## 🏋️ Training

To train AquaStereo with a single GPU, run:

```bash
python train.py
```

For distributed training, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ddp.py
```

or use `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ddp.py
```

Please modify dataset paths, batch size, training schedule, and checkpoint paths according to your local environment.

---

## 📁 Project Structure

```text
AquaStereo-main/
├── core/
├── dinov2/
├── fig/
│   ├── network.png
│   └── teaser.png
├── pretrained/
├── evaluate_stereo.py
├── train.py
├── train_ddp.py
├── README.md
└── requirements.txt
```

---

## 📌 Notes

* Large model weights are not included in this repository.
* Please download pretrained weights from Hugging Face.
* Dataset files should be prepared manually.
* Checkpoints, logs, and output files are recommended to be excluded from Git tracking.

Recommended `.gitignore` rules:

```gitignore
# model weights and checkpoints
*.pth
*.pyth
*.pt
*.ckpt
*.safetensors
*.onnx

pretrained/
checkpoints/
ckpts/
weights/
outputs/
output/
logs/
runs/
wandb/

# python cache
__pycache__/
*.pyc
```

---

## 📝 Citation

If you find AquaStereo useful in your research, please consider citing our work:

```bibtex
@inproceedings{aquastereo,
  title     = {AquaStereo},
  author    = {},
  booktitle = {},
  year      = {}
}
```

---

## 🙏 Acknowledgements

This project is built upon several excellent open-source stereo matching and vision foundation model projects. We sincerely thank the authors and contributors of these works for their valuable contributions to the community.

---

## 📬 Contact

For questions or discussions, please contact:

```text
YOUR_EMAIL_HERE
```
