# NeRFCodec
This is the official reference implementation of the paper "NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation."

## Acknowledgement
Our work builds upon excellent open-source contributions in the NeRF and Neural Compression fields:
- [TensoRF](https://github.com/apchenstu/TensoRF)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) 

Please adhere to their licenses. We thank the authors of these outstanding works and their repositories.

## Installation

First, install environment (which is basically same as TensoRF):
```
conda create -n NeRFCodec python=3.8
conda activate NeRFCodec
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
pip install matplotlib plyfile pytorch_msssim
```


## Training 
**Single scene training**
1. Pre-train the TensoRF with 30,000 iterations:
``` bash
bash scripts/tensorf_train/run_chair.sh
```
2. Warm up the modified neural codec and proceed with the joint training of the neural feature codec and the neural radiance field for 100,000 iterations:
``` bash 
bash scripts/joint_training/run_chair_feat_codec.sh # low rate point
bash scripts/joint_training/run_chair_feat_codec_384.sh # high rate point
```

**Multiple scene parallel training**
We reuse the training script ["auto_run_paramsets.py"](extra/auto_run_paramsets.py) from [TensoRF](https://github.com/apchenstu/TensoRF) project for the execution of parallel training.

## Evaluation
To be released...

## Contact
If you have any issue regrading this repository, e.g. environment setup, training scripts, more technical details, please e-mail [Sicheng Li](jasonlisicheng@zju.edu.cn). 

## Citation
If you find our code or paper useful, please cite
```tex
@InProceedings{Li_2024_NeRFCodec,
    author    = {Li, Sicheng and Li, Hao and Liao, Yiyi and Yu, Lu},
    title     = {NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-efficient Scene Representation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```
