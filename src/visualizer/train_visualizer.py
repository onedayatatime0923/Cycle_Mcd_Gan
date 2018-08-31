
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
            self.data[name] += data[i]
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
        message += 'Loss:\n'
        for name in self.data:
            message += '{:>25}: {:.4f}\n'.format(name, self.data[name]/ self.nStep)
        if len(data) > 0:
            message += 'Accu, mIOU:\n'
            for i in data:
                name = i.replace('miou','')
                message += '{:>25}: Accu: {:>8.4f}% | mIOU: {:>8.4f}\n'.format(name, *data[i])
        print(message)
        self.reset()
