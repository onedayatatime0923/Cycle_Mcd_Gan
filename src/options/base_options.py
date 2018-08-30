
import argparse, os
from ast import literal_eval
import torch


class BaseOptions():
    def __init__(self):
        self.parser = None
        self.opt = None
        self.message = None

    def initialize(self, parser):
        # ---------- Define Mode ---------- #
        parser.add_argument('--mode', type=str, choices = ['train','test'], default = 'train',
                            help="Model Mode")
        # ---------- Define Network ---------- #
        parser.add_argument('--gpuIds', type=int, nargs = '+', default=[0], help='gpu ids: e.g. 0, 0 1, 0 1 2,  use -1 for CPU')
        parser.add_argument('--model', type=str, choices = ['mcd','cycle_mcd', 'cycle_gan', 'test'], default = 'mcd',
                            help="Method Name")
        parser.add_argument('--pretrained', action = 'store_true', help='whether to use pretrained model')
        parser.add_argument('--pretrainedRoot', type = str, default = 'pretrained/', help='path to load pretrained model')
        # ---------- Optimizers ---------- #
        parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd',
                            help="network optimizer")
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum sgd (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=2e-5,
                            help='weight_decay (default: 2e-5)')
        parser.add_argument("--adjustLr", action="store_true",
                            help='whether you change lr')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--batchSize', type=int, default=1,
                            help="batch_size")
        parser.add_argument('--epoch', type=int, default=1,
                            help='the training epoch.')
        parser.add_argument('--nEpochStart', type=int, default=1,
                            help='# of epoch at starting learning rate')
        parser.add_argument('--nEpochDecay', type=int, default=1,
                            help='# of epoch to linearly decay learning rate to zero')
        # ---------- Optional Hyperparameters ---------- #
        parser.add_argument('--augment', action="store_true",
                            help='whether you use data-augmentation or not')
        parser.add_argument('--loadSize', type=int,
                            default=(512, 1024), nargs=2, metavar=("H", "W"),
                            help="H W")
        parser.add_argument('--fineSize', type=int,
                            default=(512, 1024), nargs=2, metavar=("H", "W"),
                            help="H W")
        parser.add_argument('--rotate', type=int,
                            default=10,
                            help="angle")
        # ---------- Input Image Setting ---------- #
        parser.add_argument("--inputCh", type=int, default=3,
                            choices=[1, 3, 4])
        parser.add_argument("--nClass", type=int, default=20)
        # ---------- Whether to Resume ---------- #
        parser.add_argument("--resume", action = 'store_true',
                            help="whether to resume")
        parser.add_argument("--resumeName", type=str, default='latest',
                            help="model(pth) path, set to latest to use latest cached model")
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--name', type=str,default = 'mcd',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                            help='models are saved here')
        parser.add_argument('--verbose', action='store_true', 
                            help='if specified, print more debugging information')
        parser.add_argument('--displayWidth', type=int, default=1,
                            help='frequency of showing training results on screen')
        parser.add_argument('--displayInterval', type=int, default=5,
                            help='frequency of showing training results on screen')
        parser.add_argument('--saveLatestInterval', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--saveEpochInterval', type=int, default=5, 
                            help='frequency of saving checkpoints at the end of epochs')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
                description='PyTorch Segmentation Adaptation')

        parser = self.initialize(parser)
        self.parser = parser
        self.opt = parser.parse_args()

    def construct_checkpoint(self,creatDir = True):
        if creatDir:
            index = 0
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            while os.path.exists(path):
                path = os.path.join(self.opt.checkpoints_dir, '{}_{}'.format(self.opt.name,index))
                index += 1
            self.opt.expPath = path
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            os.makedirs(self.opt.expPath)
            os.makedirs(self.opt.logPath)
            os.makedirs(self.opt.modelPath)
        else:
            self.opt.expPath = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            assert( os.path.exists(self.opt.expPath) and os.path.exists(self.opt.logPath) and os.path.exists(self.opt.modelPath) )

    def construct_outputPath(self,creatDir = True):
        self.opt.outputPath = os.path.join(self.opt.expPath, 'output')
        if not os.path.exists(self.opt.outputPath) :
            os.makedirs(self.opt.outputPath)

    def load_options(self, path):
        # load from the disk
        file_name = os.path.join(self.opt.expPath, path)
        with open(file_name, 'rt') as opt_file:
            for line in opt_file:
                if line == '-------------------- Options ------------------\n' or \
                   line == '-------------------- End ----------------------\n':
                       continue
                line = line.split('[default: ',1)[0].strip()
                arg, val = line.split(': ',1)
                # only resume has None type so yet it would't be saved
                if hasattr(self.opt, arg):
                    valType = type(getattr(self.opt, arg))
                    if (valType == list) or (valType == tuple) or (valType == bool):
                        setattr(self.opt, arg, literal_eval(val))
                    else:
                        setattr(self.opt, arg, (type(getattr(self.opt, arg)))(val))

    def construct_message(self):
        message = ''
        message += '-------------------- Options ------------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------- End ----------------------'
        self.message = message

    def save_options(self, path):
        # save to the disk
        file_name = os.path.join(self.opt.expPath, path)
        with open(file_name, 'wt') as opt_file:
            opt_file.write(self.message)
            opt_file.write('\n')
        
    def print_options(self):
        print(self.message)

    def construct_device(self):
        # set gpu ids
        if self.opt.gpuIds[0] != -1:
            self.opt.device = torch.device(self.opt.gpuIds[0])
        else:
            self.opt.device = torch.device('cpu')

    def test(self):
        pass

    def parse(self):
        # gather options
        self.gather_options()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.construct_checkpoint(creatDir = True)
        elif self.opt.mode == 'train' and self.opt.resume:
            self.construct_checkpoint(creatDir = False)
        elif self.opt.mode == 'test':
            self.construct_checkpoint(creatDir = False)
            self.construct_outputPath()

        # continue to train
        if self.opt.mode == 'train' and self.opt.resume:
            self.load_options('opt.txt')

        # print options
        self.construct_message()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.save_options('opt.txt')
        if self.opt.mode == 'test':
            self.save_options('test_opt.txt')
        self.print_options()

        # set gpu ids
        self.construct_device()

        return self.opt

    def update(self):
        self.construct_message()
        self.save_options()

