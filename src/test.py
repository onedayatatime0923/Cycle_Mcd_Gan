

import numpy as np
import matplotlib.pyplot as plt
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose
# Custom includes
from evaluator import Evaluator
from options import TestOptions
from dataset import createDataset
from models import createModel
from utils import Resize, ToTensor, Normalize, Colorize, Denormalize
from visualizer import TestVisualizer

assert torch and Variable

def plot_im(im):
    im = (Denormalize()(im).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    plt.imshow(im)
    plt.show()
def plot_seg(im):
    im = (Colorize()(im).squeeze(0).numpy()).astype(np.uint8)
    plt.imshow(im)
    plt.show()

opt = TestOptions().parse()

# set dataloader

transformList = [
    Resize(opt.loadSize),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
]

transform = Compose(transformList)

dataset = createDataset(opt.dataset,
        transform= transform, outputFile = True)[0]

dataLoader= torch.utils.data.DataLoader(
    dataset,
    batch_size= opt.batchSize, shuffle=False)

# set model

model = createModel(opt)
model.setup(opt)
model.eval()

# set visualizer
visualizer = TestVisualizer(opt, dataset)

# set evaluator
evaluator = Evaluator(opt)

for i, (data, path) in enumerate(dataLoader):

    model.set_input(data)
    model.forward()

    accu, miou = evaluator(path, model.pred)
    data = {'Accu': accu, 'mIOU': miou}
    visualizer('Val', 0, data)
visualizer.end('Val', 0, data)


