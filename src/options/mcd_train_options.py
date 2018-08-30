
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class McdTrainOptions(BaseOptions):
    def __init__(self):
        super(McdTrainOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode= 'train')
        # ---------- Define Network ---------- #
        parser.set_defaults(model= 'mcd')
        parser.add_argument('--net', type=str, default="drn_d_38", help="network structure",
                            choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                                     "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                                     "drn_d_38", "drn_d_54", "drn_d_105"])
        parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
                            choices=["18", "34", "50", "101", "152"], help='which resnet 18,50,101,152')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--sourceDataset', type=str, nargs = '+', choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = ["gta_train"])
        parser.add_argument('--targetDataset', type=str, nargs = '+',
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = ["city_train"])
        # ---------- Optimizers ---------- #
        parser.set_defaults(opt= 'sgd')
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
        parser.set_defaults(epoch= 1)
        parser.set_defaults(nEpochStart= 10)
        parser.set_defaults(nEpochDecay= 10)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'mcd_da')
        parser.set_defaults(displayWidth= 3)

        return parser
