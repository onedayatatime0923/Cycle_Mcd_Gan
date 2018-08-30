
import sys
sys.path.append('./')
from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    def __init__(self):
        self.mode = "test"
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode= 'test')
        # ---------- Define Network ---------- #
        parser.set_defaults(model= 'test')
        parser.add_argument('--net', type=str, default="drn_d_38", help="network structure",
                            choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                                     "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                                     "drn_d_38", "drn_d_54", "drn_d_105"])
        parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
                            choices=["18", "34", "50", "101", "152"], help='which resnet 18,50,101,152')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--dataset', type=str, nargs = '+', choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = ["city_val"])
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'mcd_da')
        parser.set_defaults(displayWidth= 3)
        return parser
