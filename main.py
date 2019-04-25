import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import shutil
from models import VariationalAutoencoder
import visdom
from datetime import datetime
import numpy as np

ROOT = './data'
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

BATCH_SIZE = 64
HIDDEN_DIM = 256
Z_DIM = 128
LR = 0.0001
INPUT_DIM = 784
EPOCHS = 100

vis = visdom.Visdom(port=8097)
now = datetime.now()
EXP_NAME = "exp-{}".format(now)


class Flatten(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return sample.view(1, -1).squeeze()

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


if __name__ == '__main__':
    trans = transforms.Compose([transforms.ToTensor(), Flatten(INPUT_DIM)])
    train_set, test_set = load_dataset(ROOT, trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

    device = torch.device("cpu")

    model = VariationalAutoencoder(INPUT_DIM, HIDDEN_DIM, Z_DIM, device)
    model.to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_loss = []

    for epoch in range(EPOCHS):
        epoch_loss = []
        epoch_z = []
        epoch_y = []
        with torch.set_grad_enabled(True):
            model.train()
            for batch_idx, (x, y) in enumerate(train_loader):

                x_hat, z, mu, log_sigma = model.forward(x)
                loss = model.loss_function(x, x_hat, mu, log_sigma)

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 30.)
                optimizer.step()

                epoch_loss.append(loss.item())
                epoch_z.extend(z.detach().cpu().numpy().tolist())
                epoch_y.extend(y.detach().cpu().numpy().tolist())

        iter_loss = np.mean(epoch_loss)
        training_loss.append(np.mean(epoch_loss))
        print("iter: {} \t loss: {}".format(epoch, iter_loss))

        vis.line(
            Y=np.array(training_loss),
            X=np.array(range(0, epoch + 1)),
            opts=dict(
                legend=["loss"],
                title="training loss",
                showlegend=True),
            win="win:train-{}-{}".format("VAE", EXP_NAME))


        if epoch % 20 == 0 :
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, True,
                path=os.path.join(ROOT, model.name),
                filename="checkpoint{}.pth.tar".format(epoch),
                version=0
            )