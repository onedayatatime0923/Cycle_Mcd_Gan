
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class CycleMcdTrainOptions(BaseOptions):
    def __init__(self):
        super(CycleMcdTrainOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode= 'train')
        # ---------- Define Network ---------- #
        parser.set_defaults(model= 'cycle_mcd')
        parser.add_argument('--segNet', type=str, default="drn_d_38", help="network structure",
                            choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                                     "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                                     "drn_d_38", "drn_d_54", "drn_d_105"])
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--dropout', action='store_false', help='no dropout for the generator')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization, default CycleGAN did not use dropout')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--supervisedADataset', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "gta_train")
        parser.add_argument('--unsupervisedADataset', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "gta_val")
        parser.add_argument('--supervisedBDataset', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "city_val")
        parser.add_argument('--unsupervisedBDataset', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "city_train")
        parser.add_argument('--domainA', type=str, default="GTA", choices = ['GTA','City'],
                            help="Domain A Name")
        parser.add_argument('--domainB', type=str, default="City", choices = ['GTA','City'],
                            help="Domain B Name")
        # ---------- Optimizers ---------- #
        parser.add_argument('--cycleOpt', type=str, default="adam", choices=['sgd', 'adam'],
                            help="cycle gan network optimizer")
        parser.add_argument('--mcdOpt', type=str, default="sgd", choices=['sgd', 'adam'],
                            help="mcd network optimizer")
        parser.set_defaults(adjustLr = True)
        # ---------- Train Details ---------- #
        parser.add_argument('--k', type=int, default=4,
                            help='how many steps to repeat the generator update')
        parser.add_argument("--nTimesDLoss", type=int, default=1)
        parser.add_argument("--bgLoss", action= "store_true",
                            help='whether you add background loss')
        parser.add_argument('--dLoss', type=str, default="diff",
                            choices=['mysymkl', 'symkl', 'diff'],
                            help="choose from ['mysymkl', 'symkl', 'diff']")
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--lsgan', action='store_false', help='do not use least square GAN, if specified, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lambdaA', type=float, default=10.0,
                            help='weight for cycle loss (A -> A -> A)')
        parser.add_argument('--lambdaB', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambdaIdentity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.set_defaults(epoch= 1)
        parser.set_defaults(nEpochStart= 10)
        parser.set_defaults(nEpochDecay= 10)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'cycle_mcd_da')
        parser.set_defaults(displayWidth= 4)

        return parser
