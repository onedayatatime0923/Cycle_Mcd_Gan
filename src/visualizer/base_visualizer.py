
import collections
import torch
import torchvision.utils
from utils import Timer
from tensorboardX import SummaryWriter

class BaseVisualizer():
    def __init__(self, stepSize, totalSize, logPath = None, displayWidth=1):
        self.timer = Timer()
        self.nStep = None
        self.stepSize = stepSize
        self.totalSize = totalSize
        # set data
        self.data = None
        # set writer
        if logPath is not None:
            self.writer = SummaryWriter(logPath)
        self.displayWidth = displayWidth
        # reset
        self.reset()
    def reset(self):
        self.timer.reset()
        self.nStep = 0
        self.data = collections.defaultdict(int)
        return self
    def __call__(self, name, epoch, data= []):
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|'.format( 
                    name, epoch , self.nStep * self.stepSize, self.totalSize,
                    100. * self.nStep * self.stepSize / self.totalSize)
        for i in data:
            name = i #.replace('data','')
            self.data[i] += data[i]
            message += ' {}:{:.4f}'.format(name, data[i])
        self.nStep += 1
        message += '|Time: {}'.format(
                self.timer(((self.nStep * self.stepSize)+ 1E-3) / self.totalSize))
        print(message, end = '', flush = True)
    def end(self, name, epoch, data): 
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|Time: {}\n'.format( 
                    name, epoch , self.totalSize, self.totalSize, 100.,
                    self.timer(1))
        for name in self.data:
            message += '{:>20}: {:.4f}\n'.format(name, self.data[name]/ self.nStep)
        for name in data:
            message += '{:>20}: {:.4f}\n'.format(name, data[name])
        print(message)
        self.reset()
    def displayScalor(self, data, step):
        for i in data:
            self.writer.add_scalar(i, data[i] ,step)
    def displayImage(self, data, step, name = 'Image'):
        image = []
        for name in data:
            im = data[name].cpu().unsqueeze(0)
            image.append(im)
        image= torchvision.utils.make_grid(torch.cat(
            image, 0), nrow = self.displayWidth, normalize = True, range=(0,1))
        self.writer.add_image(name , image, step)
