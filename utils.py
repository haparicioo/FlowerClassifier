from torchvision import models
from torch import nn, optim

import warnings
# Dont display warning for CUDA drivers
warnings.filterwarnings("ignore", category=UserWarning)

import torch, sys


DEVICE = 'cpu'
SUPPORTED_ARCHS = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'dn121': models.densenet121,
    'dn161': models.densenet161,
    'dn169': models.densenet169,
    'dn201': models.densenet201,
}


def getModel(arch, hidden_units, output_units, drop=0.2):
    model = SUPPORTED_ARCHS.get(arch, models.vgg11)(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    input_units = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=drop),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=drop),
        nn.Linear(hidden_units, output_units),
        nn.LogSoftmax(dim=1)
    )
    model.classifier.arch = arch
    model.classifier.input_units = input_units
    model.classifier.hidden_units = hidden_units
    model.classifier.output_units = output_units

    return model

def save_checkpoint(model, optimizer, epoch, save_dir):
    if not save_dir.exists():
        print(f"ERROR: save_dir \"{save_dir}\" does not exists", file=sys.stderr)
        exit()

    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'model_arch': model.arch,
        'model_input_units': model.input_units,
        'model_hidden_units': model.hidden_units,
        'model_output_units': model.output_units,
    }
    torch.save(checkpoint, save_dir.joinpath('model.pth'))
    print(f"Checkpoint saved")

def load_checkpoint(save_dir, learning_rate=0.002):
    if save_dir.is_dir():
        file_name = save_dir.joinpath('model.pth')
    elif save_dir.is_file():
        file_name = save_dir

    if not file_name.exists():
        print(f"WARNING: checkpoint \"{file_name}\" does not exists", file=sys.stderr)
        return None, None, None

    checkpoint = torch.load(file_name)
#   checkpoint = torch.load(file_name, map_location=DEVICE)

    arch = checkpoint['model_arch']
    input_units  = checkpoint['model_input_units']
    hidden_units = checkpoint['model_hidden_units']
    output_units = checkpoint['model_output_units']

    model = getModel(arch, hidden_units, output_units)
    model.to(DEVICE)

    model.classifier.load_state_dict(checkpoint['model_state'])
    model.classifier.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    return model, optimizer, checkpoint['epoch']

