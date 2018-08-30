
import os
from os.path import basename
from PIL import Image
import numpy as np
import torch
from utils import IouEval, LabelFilter
from utils import palette20, palette35


class Evaluator:
    def __init__(self, opt):
        self.opt = opt
        self.labelFilter = LabelFilter()
        self.iouEval = IouEval(opt.nClass, ignoreEnd = True )
    def __call__(self, path, predImage):
        predImage = predImage.cpu()
        imageFile, labelFile = path
        for i in range(len(imageFile)):
            image = Image.open(imageFile[i])
            pred = Image.fromarray(predImage[i].numpy().astype(np.uint8)).convert('P')
            pred.putpalette(palette20.palette)
            pred =  pred.resize(image.size, Image.NEAREST)
            gnd = Image.open(labelFile[i]).convert('P')
            gnd.putpalette(palette35.palette)
            # for iou caculating
            predTensor = torch.LongTensor(np.array(pred))
            gndTensor = self.labelFilter(torch.LongTensor(np.array(gnd)))
            
            image.save(os.path.join(self.opt.outputPath, basename(imageFile[i])))
            pred.save(os.path.join(self.opt.outputPath, 'pred.'.join(basename(imageFile[i]).split('.'))))
            gnd.save(os.path.join(self.opt.outputPath, 'gnd.'.join(basename(imageFile[i]).split('.'))))

            self.iouEval.update(predTensor, gndTensor)

        return self.iouEval.metric()

            


