
import torch
from collections import OrderedDict
from utils import Colorize
from models.drn.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
from models.base_model import BaseModel


class TestModel(BaseModel):
    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        print('-------------- Networks initializing -------------')

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.lossNames = []
        # specify the training miou you want to print out. The program will call base_model.get_current_losses
        self.miouNames = []

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # only image doesn't have prefix
        self.imageNames = ['image', 'pred', 'gnd']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # naming is by the input domain
        self.modelNames = ['net{}'.format(i) for i in 
                ['Features', 'Classifier1', 'Classifier2']]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_RGB (G), G_D (F), D_RGB (D_Y), D_D (D_X)
        self.netFeatures = self.initNet(DRNSegBase(model_name=opt.net, 
                n_class=opt.nClass, input_ch=opt.inputCh))
        self.netClassifier1 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))
        self.netClassifier2 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))

        self.set_requires_grad([self.netFeatures, self.netClassifier1, 
            self.netClassifier2], True)

        self.colorize = Colorize()
        print('--------------------------------------------------')
    def name(self):
        return 'TestModel'

    def current_images(self):
        visual_ret = OrderedDict()
        for name in self.imageNames:
            visual_ret[name] = \
            self.colorize(getattr(self, name)[0]).permute(2,0,1).float()/255
        return visual_ret

    def set_input(self, input):
        self.image = input['image'].to(self.opt.device)
        self.gnd = input['label'].to(self.opt.device)

    def forward(self):
        with torch.no_grad():
            feature = self.netFeatures(self.image)
            pred1 = self.netClassifier1(feature)
            pred2 = self.netClassifier2(feature)

            self.pred = (pred1 + pred2).argmax(1)

