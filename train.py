from torchvision import transforms, datasets, models
from torch.utils import data
from torch import nn, optim
from os import path

import json, torch, argparse, pathlib, sys, warnings, time
import numpy as np


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train CNN on a given dataset')
    arg_parser.add_argument('data_dir',
        help='Training data with /train /valid /test subfolders',
        nargs=1,
        type=pathlib.Path,
    )
    arg_parser.add_argument('--save_dir',
        help='Directory where to save checkpoints (it must exist)',
        nargs='?',
        type=pathlib.Path,
        default='.',
    )
    arg_parser.add_argument('--arch',
        help='Network architecture for feature detection. For details see https://pytorch.org/docs/stable/torchvision/models.html',
        nargs='?',
        default='vgg11',
    )
    arg_parser.add_argument('--learning_rate',
        help='Optimizer learning rate',
        nargs='?',
        type=float,
        default=0.002,
    )
    arg_parser.add_argument('--hidden_units',
        help='Classifier hidden units number',
        nargs='?',
        type=int,
        default=4096,
    )
    arg_parser.add_argument('--epochs',
        help='Number of epochs for model training',
        nargs='?',
        type=int,
        default=5,
    )
    arg_parser.add_argument('--gpu',
        help='Use gpu for speed-up learning process',
        action='store_true',
    )

    ARGS = arg_parser.parse_args()
