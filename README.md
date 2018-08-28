
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
python3 datamanager/split_gta.py [directory of GTA Dataset] [path of split.mat]
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
- Generate txt file
```
python3 datamanager/generate_txt.py [directory of GTA Dataset]
```
### Train
- Train a model:
```bash
python3 cycle_mcd_trainer.py
```
## Display UI
Optionally, for displaying images during training and test, use the [tensorboardX](https://github.com/lanpa/tensorboardX)
```bash
cd checkpoints/cycle_mcd_da
tensorboard --logdir log
```
## Citation

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

```
```
@article{saito2017maximum,
  title={Maximum Classifier Discrepancy for Unsupervised Domain Adaptation},
  author={Saito, Kuniaki and Watanabe, Kohei and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv preprint arXiv:1712.02560},
  year={2017}
}

```


## Related Projects:
[pix2pix](https://github.com/phillipi/pix2pix): Image-to-image translation using conditional adversarial nets  
[iGAN](https://github.com/junyanz/iGAN): Interactive Image Generation via Generative Adversarial Networks

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper Collection:  
[[Github]](https://github.com/junyanz/CatPapers) [[Webpage]](http://people.eecs.berkeley.edu/~junyanz/cat/cat_papers.html)

## Acknowledgments


## Related Projects:

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004v1.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>
`kdjflk`

