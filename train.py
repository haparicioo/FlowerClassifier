from torchvision import transforms, datasets, models
from torch.utils import data
from torch import nn, optim
from os import path

import json, torch, argparse, pathlib, sys, warnings, time
import numpy as np

# Custom module
import utils


def getDataLoaders(data_dir):
    train_dir = data_dir.joinpath('train')
    valid_dir = data_dir.joinpath('valid')
    test_dir  = data_dir.joinpath('test')

    if not data_dir.exists():
        print(f"ERROR: data_dir \"{data_dir}\" does not exists", file=sys.stderr)
        exit()

    if not train_dir.exists():
        print(f"ERROR: train_dir \"{train_dir}\" does not exists", file=sys.stderr)
        exit()

    if not valid_dir.exists():
        print(f"ERROR: valid_dir \"{valid_dir}\" does not exists", file=sys.stderr)
        exit()

    if not test_dir.exists():
        print(f"ERROR: test_dir \"{test_dir}\" does not exists", file=sys.stderr)
        exit()

    train_transforms = transforms.Compose([
        transforms.RandomRotation(35),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_valid_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    dataset_train = datasets.ImageFolder(train_dir, train_transforms)
    dataset_test  = datasets.ImageFolder(test_dir, test_valid_transforms)
    dataset_valid = datasets.ImageFolder(valid_dir, test_valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    loaders={}
    loaders['train'] = data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    loaders['valid'] = data.DataLoader(dataset_valid, batch_size=64, shuffle=True)
    loaders['test']  = data.DataLoader(dataset_test , batch_size=64, shuffle=True)

    return loaders


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
