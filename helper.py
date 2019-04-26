import torchvision.datasets as dset
import torch
import shutil
import os
import numpy as np
class Flatten(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return sample.view(1, -1).squeeze()

class Round(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.round(sample)



def load_dataset(root, transformation, download=True):

    # if not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=transformation, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=transformation, download=download)

    return train_set, test_set

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar', version=0):
    torch.save(state, ensure_dir(os.path.join(path, version, filename)))
    if is_best:
        shutil.copyfile(os.path.join(path, version, filename), os.path.join(path, str(version), 'model_best.pth.tar'))

def ensure_dir(file_path):
    '''
    Used to ensure the creation of a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path