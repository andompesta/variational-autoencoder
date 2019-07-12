import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, image_channels=1, output_channels=[32, 64, 128, 256],  h_dim=512, z_dim=32, device=torch.device("cpu")):
        super(CNNVariationalAutoencoder, self).__init__()
        self.name = "CNNVariationalAutoencoder"

        assert isinstance(output_channels, list)
        assert len(output_channels) == 4

        self.hidden_dim = h_dim
        self.latent_dim = z_dim
        self.device = device

        self.std_normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(z_dim),
                                                                                          torch.eye(z_dim),
                                                                                          )
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, output_channels[0], kernel_size=3, stride=2),
            nn.BatchNorm2d(output_channels[0]),
            nn.ReLU(),
            nn.Conv2d(output_channels[0], output_channels[1], kernel_size=3, stride=2),
            nn.BatchNorm2d(output_channels[1]),
            nn.ReLU(),
            nn.Conv2d(output_channels[1], output_channels[2], kernel_size=3, stride=2),
            nn.BatchNorm2d(output_channels[2]),
            nn.ReLU(),
            # nn.Conv2d(output_channels[2], output_channels[3], kernel_size=4, stride=2),
            # nn.BatchNorm2d(output_channels[3]),
            # nn.ReLU()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(h_dim, output_channels[2], kernel_size=5, stride=2),
            # nn.ReLU(),
            nn.ConvTranspose2d(h_dim, output_channels[1], kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels[1], output_channels[0], kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels[0], image_channels, kernel_size=4, stride=2)
        )


    def flatten(self, x):
        return x.view(x.size(0), -1)

    def unflatten(self, x, size=None):
        if size == None:
            size = self.hidden_dim

        return x.view(x.size(0), size, 1, 1)


    def sample_from_prior(self, z_sample):
        batch_size, _ = z_sample.size()
        p_x_given_z_prior_logits = self.decoder_net(z_sample)
        p_x_given_z_sample = torch.distributions.bernoulli.Bernoulli(logits=p_x_given_z_prior_logits).sample()
        return p_x_given_z_sample

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h_flat = self.flatten(h)
        z, mu, logvar = self.bottleneck(h_flat)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z_unflat = self.unflatten(z, self.hidden_dim)
        p_x_given_z_logits = self.decoder(z_unflat)
        x_hat = torch.distributions.bernoulli.Bernoulli(logits=p_x_given_z_logits).sample().to(self.device)
        # x_hat = F.sigmoid(p_x_given_z_logits)
        return x_hat, p_x_given_z_logits

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat, p_x_given_z_logits = self.decode(z)
        return x_hat, p_x_given_z_logits, z, mu, logvar

    def loss_function(self, x, x_hat_logit, mu, log_sigma):
        """
        Not sure on how to sample the negative edges (they should be the anomalous edges).
        Potentialy using the penalty weight should work well. However, just ry to minimise the neg-log likelihood
        :return:
        """
        rec_loss = nn.functional.binary_cross_entropy_with_logits(x_hat_logit, x, size_average=False)
        kl_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        return rec_loss + (5 * kl_loss), rec_loss, kl_loss

    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        for p in self.parameters():
            if len(p.shape) > 1 and p.size(-1) != 2:
                nn.init.xavier_normal_(p)

class VariationalAutoencoder(torch.nn.Module):
    """
    PyTorch implementation of the GCMN described in: https://paypal.box.com/s/p2kji0qxyeseq31lko25z7qni5pek9ss
    """
    def __init__(self, input_dim, hidden_dim, z_dim, device):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.name = "VariationalAutoencoder"


        self.device = device
        self.eps = 1e-12


        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim)
        )


        self.decoder_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()      # it is similar to a bernoulli prob distribution
        )

        self.device = device
        self.std_normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(z_dim),
                                                                                          torch.eye(z_dim),
                                                                                          )
        # self.x_dist = torch.distributions.bernoulli.Bernoulli()


    def sample_from_prior(self, z_sample):
        batch_size, _ = z_sample.size()
        p_x_given_z_prior_logits = self.decoder_net(z_sample)
        p_x_given_z_sample = torch.distributions.bernoulli.Bernoulli(logits=p_x_given_z_prior_logits).sample()
        return p_x_given_z_sample


    def forward(self, x):
        """
        forward step
        """
        batch_size, _ = x.size()

        params = self.encoder_net(x)
        mu, log_sigma = torch.chunk(params, 2, dim=-1)

        std_normal_sample = self.std_normal_dist.sample((batch_size,)).to(self.device)

        z = mu + (torch.exp(log_sigma) * std_normal_sample)

        p_x_given_z_logits = self.decoder_net(z)
        p_x_given_z = torch.sigmoid(p_x_given_z_logits)
        x_hat = torch.bernoulli(p_x_given_z)

        # print(np.unique(x.detach().cpu().numpy()))
        # print(np.unique(x_hat.detach().cpu().numpy()))

        return x_hat, p_x_given_z_logits, z, mu, log_sigma

    def loss_function(self, x, p_x_given_z_logits, mu, log_sigma):
        """
        Not sure on how to sample the negative edges (they should be the anomalous edges).
        Potentialy using the penalty weight should work well. However, just ry to minimise the neg-log likelihood
        :return:
        """
        rec_loss = nn.functional.binary_cross_entropy_with_logits(p_x_given_z_logits, x, reduction="none").sum(dim=1)
        kl_loss = 0.5 * torch.sum(torch.exp(log_sigma) + mu**2 - log_sigma - 1., dim=1)

        return torch.mean(rec_loss + kl_loss), rec_loss.mean(), kl_loss.mean()


    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        for p in self.parameters():
            if len(p.shape) > 1 and p.size(-1) != 2:
                nn.init.xavier_normal_(p)