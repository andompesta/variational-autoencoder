import torch
from torch import nn
from torch import functional as F

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
            nn.Sigmoid()      # it is similar to a bernoulli prob distribution
        )

        self.device = device
        self.std_normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim), device=self.device)
        # self.x_dist = torch.distributions.bernoulli.Bernoulli()

    def forward(self, x):
        """
        forward step
        """
        batch_size, _ = x.size()

        params = self.encoder_net(x)
        mu, log_sigma = torch.chunk(params, 2, dim=-1)

        std_normal_sample = self.std_normal_dist.sample_n(batch_size)

        z = mu + (torch.exp(log_sigma) * std_normal_sample)

        x_hat = self.decoder_net(z)


        return x_hat, z, mu, log_sigma

    def loss_function(self, x, x_hat, mu, log_sigma):
        """
        Not sure on how to sample the negative edges (they should be the anomalous edges).
        Potentialy using the penalty weight should work well. However, just ry to minimise the neg-log likelihood
        :return:
        """
        rec_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1)
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