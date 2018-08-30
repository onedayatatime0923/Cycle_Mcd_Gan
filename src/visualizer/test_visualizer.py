
from visualizer.base_visualizer import BaseVisualizer

class TestVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        super(TestVisualizer, self).__init__(opt.batchSize, len(dataset))
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
        for name in data:
            message += '{:>10}: {:.4f}\n'.format(name, data[name])
        print(message)
        self.reset()
