

import numpy as np
import matplotlib.pyplot as plt
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose
# Custom includes
from visualizer import TrainVisualizer
from options import CycleGanTrainOptions
from dataset import SourceTargetDataset, createDataset
from models import createModel
from utils import RandomRotation, RandomCrop, Resize, ToTensor, Normalize, RandomHorizontalFlip, Colorize, Denormalize

assert torch and Variable

def plot_im(im):
    im = (invTransform(im).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    plt.imshow(im)
    plt.show()
def plot_seg(im):
    im = (Colorize()(im).squeeze(0).numpy()).astype(np.uint8)
    plt.imshow(im)
    plt.show()

opt = CycleGanTrainOptions().parse()
invTransform = Denormalize(opt.normalizeMean, opt.normalizeNorm)

# set model

model = createModel(opt)
model.setup(opt)

# set dataloader

if opt.augment:
    transformList = [
        Resize(opt.loadSize),
        RandomCrop(opt.fineSize),
        RandomRotation(opt.rotate),
        ToTensor(),
        Normalize(opt.normalizeMean,
            opt.normalizeNorm),
        RandomHorizontalFlip(),
    ]
else:
    transformList = [
        Resize(opt.loadSize),
        ToTensor(),
        Normalize(opt.normalizeMean,
            opt.normalizeNorm)
    ]

transform = Compose(transformList)

datasetA = createDataset([opt.datasetA],
        transform= transform, outputFile = False)

datasetB = createDataset([opt.datasetB],
        transform= transform, outputFile = False)

dataset = SourceTargetDataset(
    source = datasetA,
    target = datasetB) 

dataLoader= torch.utils.data.DataLoader(
    dataset, batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)

# set visualizer

visualizer = TrainVisualizer(opt, dataLoader.dataset).reset()

steps = 0
for epoch in range(opt.epoch, opt.nEpochStart + opt.nEpochDecay + 1):
    for i, data in enumerate(dataLoader):
        steps += 1


        model.set_input(data)
        model.optimize_parameters()

        visualizer('Train', epoch, data = model.current_losses())

        if steps % opt.displayInterval == 0:
            visualizer.displayImage(model.current_images(), steps)
            visualizer.displayScalor(model.current_losses(), steps)

        if steps % opt.saveLatestInterval == 0:
            print('\nsaving the latest model (epoch %d, total_steps %d)' % (epoch, steps))
            model.save_networks('latest')


    if epoch % opt.saveEpochInterval == 0:
        print('\nsaving the model at the end of epoch %d, iters %d' % (epoch, steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    visualizer.end('Train', epoch, data = model.current_mious())
    print('='*80)
    if opt.adjustLr:
        model.update_learning_rate()
