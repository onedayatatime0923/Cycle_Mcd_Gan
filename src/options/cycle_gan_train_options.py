
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class CycleGanTrainOptions(BaseOptions):
    def __init__(self):
        super(CycleGanTrainOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode= 'train')
        # ---------- Define Network ---------- #
        parser.set_defaults(model= 'cycle_gan')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--dropout', action='store_true', help='do not use dropout for the generator, if specified, use dropout')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization, default CycleGAN did not use dropout')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--datasetA', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "gta_train")
        parser.add_argument('--datasetB', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "city_train")
        # ---------- Optimizers ---------- #
        parser.set_defaults(opt= 'adam')
        parser.set_defaults(lr= 2E-4)
        parser.set_defaults(adjustLr= True)
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--lsgan', action='store_false', help='do not use least square GAN, if specified, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lambdaA', type=float, default=10.0,
                            help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambdaB', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambdaIdentity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.set_defaults(epoch= 1)
        parser.set_defaults(nEpochStart= 100)
        parser.set_defaults(nEpochDecay= 100)
        # ---------- Optional Hyperparameters ---------- #
        parser.set_defaults(augment= True)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'cycle_gan')
        parser.set_defaults(displayWidth= 4)

        return parser
