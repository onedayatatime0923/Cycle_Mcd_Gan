
import argparse, os
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from utils import IouEval, LabelFilter

parser = argparse.ArgumentParser(
        description='PyTorch Segmentation Adaptation')
parser.add_argument('imageDir', type=str)
parser.add_argument("--nClass", type=int, default=20)

opt = parser.parse_args()

iouEval = IouEval(opt.nClass, ignoreEnd = True)
labelFilter = LabelFilter()

files = os.listdir(opt.imageDir)
files.sort()

for f in tqdm(files):
    if f.endswith('pred.png'):
        predFile = f
        gndFile = f.replace('pred','gnd')
        predTensor = torch.LongTensor(np.array(Image.open(os.path.join(opt.imageDir,predFile))))
        gndTensor = torch.LongTensor(np.array(Image.open(os.path.join(opt.imageDir,gndFile))))
        gndTensor = labelFilter(gndTensor)

        iouEval.update(predTensor, gndTensor)

print('Accu: {}% | mIOU: {}%'.format(*iouEval.metric()))
