
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
- Clone this repo:
```bash
git clone https://github.com/onedayatatime0923/Cycle_Mcd_Gan
cd Cycle_Mcd_Gan
```

### Dataset

#### Cityscapes
- Download **gtFine_trainvaltest.zip** and **leftImg8bit_trainvaltest.zip** from https://www.cityscapes-dataset.com/downloads/
- Unzip both files
- Rename the directory as followed
```
Cityscapes
└───image
│   └───train
│   │   └───aachen
│   │   │     aachen_000000_000019_leftImg8bit.png
│   │   │     ...
│   │   ...
│   │
│   └───val
│   │   └───frankfurt
│   │   │     frankfurt_000000_000294_leftImg8bit.png
│   │   │     ...
│   │   ...
│   │
│   └───test
│       └───berlin
│       │     berlin_000000_000019_leftImg8bit.png
│       │     ...
│       ...
│   
└───label
    └───train
    │   └───aachen
    │   │     aachen_000000_000019_gtFine_labelIds.png
    │   │     ...
    │   ...
    │
    └───val
    │   └───frankfurt
    │   │     frankfurt_000000_000294_gtFine_labelIds.png
    │   │     ...
    │   ...
    │
    └───test
        └───berlin
        │     berlin_000000_000019_gtFine_labelIds.png
        │     ...
        ...
```
- Generate txt file
```
python3 datamanager/generate_txt.py [directory of Cityscapes Dataset]
```

#### GTA
- Download **all the images and labels** and **split.mat** from https://download.visinf.tu-darmstadt.de/data/from_games/
- Unzip all files
- Rename the directory as followed
```
GTA
└───image
│     aachen_000000_000019_leftImg8bit.png
│     ...
│   
└───label
      aachen_000000_000019_leftImg8bit.png
      ...
```
- Split data
```
python3 datamanager/split_gta.py [directory of Cityscapes Dataset] [path of split.mat]
```
**Note**: the datastructure will become like this

```
Cityscapes
└───image
│   └───train
│   │     aachen_000000_000019_leftImg8bit.png
│   │     ...
│   │
│   └───val
│   │     aachen_000000_000019_leftImg8bit.png
│   │     ...
│   │
│   └───test
│         aachen_000000_000019_leftImg8bit.png
│         ...
│   
└───label
    └───train
    │     aachen_000000_000019_gtFine_labelIds.png
    │     ...
    │
    └───val
    │     aachen_000000_000019_gtFine_labelIds.png
    │     ...
    │
    └───test
          aachen_000000_000019_gtFine_labelIds.png
          ...
```



**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004v1.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>
`kdjflk`

