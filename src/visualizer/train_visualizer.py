
import torch
import torchvision.utils
from visualizer.base_visualizer import BaseVisualizer

class TrainVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        super(TrainVisualizer, self).__init__(opt.batchSize, len(dataset), opt.logPath, opt.displayWidth)
        # init argument:  stepSize, totalSize, displayWidth, logPath):
    def __call__(self, name, epoch, data= []):
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|'.format( 
                    name, epoch , self.nStep * self.stepSize, self.totalSize,
                    100. * self.nStep * self.stepSize / self.totalSize)
        for i in data:
            name = i.replace('loss','')
            self.data[i] += data[i]
            message += ' {}:{:.4f}'.format(name, data[i])
        self.nStep += 1
        message += '|Time: {}'.format(
                self.timer(self.nStep * self.stepSize / self.totalSize))
        print(message, end = '', flush = True)
    def end(self, name, epoch, data): 
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|Time: {}\n'.format( 
                    name, epoch , self.totalSize, self.totalSize, 100.,
                    self.timer(1))
        for name in self.data:
            message += '{:>10}: {:.4f}\n'.format(name, self.data[name]/ self.nStep)
        for name in data:
            message += '{:>10}: {:.4f}\n'.format(name, data[name])
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
