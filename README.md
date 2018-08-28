
# Combination of CycleGan and Mcd in Pytorch

This is my PyTorch implementation for semi-supervised un-paired co-training. Although it is not yet been completed, it is nolonger under development.


This package includes [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [MCD_DA](https://github.com/mil-tokyo/MCD_DA)

The code was written by [Chia-Ming Chang](https://github.com/onedayatatime0923).

**Note**: The current software works well with PyTorch 0.4.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

## Getting Started
### Installation
- Install torch and dependencies from https://github.com/torch/distro
- Install tensorboardX from https://github.com/lanpa/tensorboardX

### Dataset

#### Cityscapes
- Download **gtFine_trainvaltest.zip** and **leftImg8bit_trainvaltest.zip** from https://www.cityscapes-dataset.com/downloads/
- Unzip

.
+-- _config.yml

+-- _drafts

|   +-- begin-with-the-crazy-ideas.textile

|   +-- on-simplicity-in-technology.markdown

+-- _includes

|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004v1.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>
`kdjflk`

