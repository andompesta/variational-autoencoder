from helper import *
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import shutil
from models import VariationalAutoencoder
import plotly.plotly as py
import plotly.graph_objs as go

from plotly import tools

from datetime import datetime
import numpy as np


ROOT = './data'
MODEL = "VariationalAutoencoder"
VERSION = "0"

DEVICE = "cuda:0"
BATCH_SIZE = 64
HIDDEN_DIM = 256
Z_DIM = 128
LR = 0.0001
INPUT_DIM = 784
EPOCHS = 5000


def generate_form_prior(model, batch_size):
    p_z = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(model.z_dim),
                                                                     torch.eye(model.z_dim))

    p_z_sample = p_z.sample((batch_size,))
    x_hat = model.sample_from_prior(p_z_sample)


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


if __name__ == '__main__':
    trans = transforms.Compose([transforms.ToTensor(), Round(), Flatten(INPUT_DIM)])
    device = torch.device(DEVICE)

    model = VariationalAutoencoder(INPUT_DIM, HIDDEN_DIM, Z_DIM, device)
    checkpoint = torch.load(os.path.join(ROOT, MODEL, VERSION, "checkpoint2200.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    train_set, test_set = load_dataset(ROOT, trans)


    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

    z_test = []
    y_test = []
    mu_test = []
    log_sigma = []

    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            x_hat, p_x_given_z, z, mu, log_sigma = model.forward(x)

            # to remove when retrained
            x_hat = torch.bernoulli(p_x_given_z)

            # loss, rec_loss, kl_div = model.loss_function(x, x_hat, mu, log_sigma)

            y_test.extend(y.cpu().numpy())
            z_test.extend(z.cpu().numpy())
            mu_test.extend(mu.cpu().numpy())

            if batch_idx % 1000 == 0:
                # plot reconstructed images
                fig = tools.make_subplots(rows=8, cols=8, print_grid=False)

                order = np.linspace(0, 63, 64)

                for img, i in zip(x_hat, order):
                    p1 = go.Heatmap(z=img.detach.cpu().numpy().reshape(28, 28), showscale=False)
                    fig.append_trace(p1, int(i / 4 + 1), int(i % 4 + 1))

                for i in map(str, range(1, 17)):
                    y = 'yaxis' + i
                    x = 'xaxis' + i
                    fig['layout'][y].update(autorange='reversed',
                                            showticklabels=False, ticks='')
                    fig['layout'][x].update(showticklabels=False, ticks='')

                fig['layout'].update(height=700)
                py.plot(fig)

