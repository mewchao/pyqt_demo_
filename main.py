import argparse
import sys

import torch
from PyQt5 import QtWidgets
from app import InteractiveDemoApp
from isegm.inference import utils
from isegm.utils import exp

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='coco_lvis_h32_itermask.pth',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=False,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=True,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    args, cfg = parse_args()
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)
    window = InteractiveDemoApp(args, model)
    window.show()
    sys.exit(app.exec_())
