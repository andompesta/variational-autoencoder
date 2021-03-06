from helper import *
import os
import torch
import torchvision.transforms as transforms
from models import VariationalAutoencoder, CNNVariationalAutoencoder
import visdom
from datetime import datetime
import numpy as np

ROOT = './data'
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

DEVICE = "cuda:0"
# DEVICE = "cpu"
BATCH_SIZE = 128
HIDDEN_DIM = 256
Z_DIM = 128
LR = 0.0001
INPUT_DIM = 784
EPOCHS = 5000

vis = visdom.Visdom(port=8097)
now = datetime.now()
EXP_NAME = "exp-{}".format(now)



if __name__ == '__main__':
    # trans = transforms.Compose([transforms.ToTensor(), Round(), Flatten(INPUT_DIM)])
    trans = transforms.Compose([transforms.ToTensor(), Round()])

    train_set, test_set = load_dataset(ROOT, trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)


    device = torch.device(DEVICE)

    model = CNNVariationalAutoencoder(device=device)
    model.to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_loss = []
    training_rec_loss = []
    training_kl_loss = []

    for epoch in range(EPOCHS):
        epoch_loss = []
        epoch_rec = []
        epoch_kl = []
        with torch.set_grad_enabled(True):
            model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                x_hat, p_x_given_z_logits, z, mu, log_sigma = model.forward(x)

                loss, rec_loss, kl_div = model.loss_function(x, p_x_given_z_logits, mu, log_sigma)
                # loss, rec_loss, kl_div = model.loss_function(x, x_hat, mu, log_sigma)
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 30.)
                optimizer.step()

                # epoch_loss.append(loss.item())
                # epoch_rec.append(rec_loss.item())
                # epoch_kl.append(kl_div.item())
                epoch_loss.append(loss.item()/BATCH_SIZE)
                epoch_rec.append(rec_loss.item()/BATCH_SIZE)
                epoch_kl.append(kl_div.item()/BATCH_SIZE)

        training_loss.append(np.mean(epoch_loss))
        training_rec_loss.append(np.mean(epoch_rec))
        training_kl_loss.append(np.mean(epoch_kl))


        vis.line(
            Y=np.column_stack([training_loss, training_rec_loss, training_kl_loss]),
            X=np.array(range(0, epoch + 1)),
            opts=dict(
                legend=["loss", "rec", "kl"],
                title="training loss",
                showlegend=True),
            win="win:train-{}-{}".format("CNN_VAE", EXP_NAME))


        if epoch % 10 == 0 :
            model.to("cpu")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()
            }, True,
                path=os.path.join(ROOT, model.name),
                filename="CNN_checkpoint{}.pth.tar".format(epoch),
                version="0"
            )
            model.to(device)