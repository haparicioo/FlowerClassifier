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

def accuracy(log_probability, categories):
    probability = torch.exp(log_probability)
    _, prediction = probability.topk(1, dim=1)
    y_yi = prediction == categories.view(*prediction.shape)

    return torch.mean(y_yi.type(torch.FloatTensor)).item()

def accuracy_on_loader(model, loader):
    loader_accuracy = 0
    model.eval()
    for images, categories in loader:
        images, categories = images.to(utils.DEVICE), categories.to(utils.DEVICE)
        log_probs = model.forward(images)
        loader_accuracy += accuracy(log_probs, categories)

    return loader_accuracy/len(loader)

def trainModel(model, loaders, learning_rate, epochs, save_dir):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


    train_loss = 0
    valid_loss = 0
    loader_step = 0
    train_accuracy = 0
    start_epochs = 0

    check_model, check_optim, check_epochs = utils.load_checkpoint(save_dir, learning_rate)
    if check_model != None:
        model = check_model
    if check_optim != None:
        optimizer = check_optim
    if check_epochs != None:
        start_epochs = check_epochs

    model.to(utils.DEVICE)

    if start_epochs == epochs:
        print(f"ERROR: number of epochs reached it's maximum {epochs}, increase epochs with --epochs option")

    for epoch in range(start_epochs, epochs):
        for images, categories in loaders['train']:
            loader_step += 1
            images, categories = images.to(utils.DEVICE), categories.to(utils.DEVICE)

            optimizer.zero_grad()

            log_train_probability = model.forward(images)
            loss = criterion(log_train_probability, categories)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy(log_train_probability, categories)

            if loader_step % epochs == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = 0, 0

                    for images, categories in loaders['valid']:
                        images, categories = images.to(utils.DEVICE), categories.to(utils.DEVICE)
                        log_probability = model.forward(images)
                        loss = criterion(log_probability, categories)

                        valid_loss = loss.item()
                        valid_accuracy += accuracy(log_probability, categories)

                len_valid_loader = len(loaders['valid'])

                print(f"Epoch {epoch+1}/{epochs} "
                      f"Train loss: {train_loss/epochs:.3f} "
                      f"Valid loss: {valid_loss/len_valid_loader:.3f} "
                      f"Train accuracy: {train_accuracy/epochs:.3f} "
                      f"Valid accuracy: {valid_accuracy/len_valid_loader:.3f} ")
                model.train()

                train_loss = 0
                train_accuracy = 0
        utils.save_checkpoint(model.classifier, optimizer, epoch+1, save_dir)
    return model


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
        choices= utils.SUPPORTED_ARCHS.keys(),
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
    #TODO: Print hyperparameters
    print(f"Training Hyperparameters")
    print(f"    Epochs: {ARGS.epochs}")
    print(f"    Learning rate: {ARGS.learning_rate}")
    print(f"    Hidden units: {ARGS.hidden_units}")
    print(f"    Model architecture: {ARGS.arch}")
    print(f"Training configuration")
    print(f"    Data directory: {ARGS.data_dir}")
    print(f"    Checkpoint directory: {ARGS.save_dir}")
    print(f"    Use GPU: {ARGS.gpu}")
    print()

    # Dont display warning for CUDA drivers
    warnings.filterwarnings("ignore", category=UserWarning)

    if ARGS.gpu:
        if torch.cuda.is_available():
            utils.DEVICE = 'cuda'
        else:
            print("WARNING: cuda is not available, using cpu instead", file=sys.stderr)
            utils.DEVICE = 'cpu'

    loaders = getDataLoaders(ARGS.data_dir[0])
    output_units = len(loaders['train'].dataset.class_to_idx)

    print(f"Training started")
    model = utils.getModel(ARGS.arch, ARGS.hidden_units, output_units)
    start = time.perf_counter()
    model.classifier.class_to_idx = loaders['train'].dataset.class_to_idx
    model = trainModel(model, loaders, ARGS.learning_rate, ARGS.epochs, ARGS.save_dir)
    stop  = time.perf_counter()
    print(f"Training finished in {stop - start:0.4f} seconds")

    test_accuracy = accuracy_on_loader(model, loaders['test'])
    print(f"Model accuracy on testing data: {test_accuracy}")

