from PIL import Image
import numpy as np
import argparse, pathlib, sys, torch, json

import utils


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    np_image = np.array(image) / 255
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))

    return torch.from_numpy(np_image).float()

