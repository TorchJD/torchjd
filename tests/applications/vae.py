import argparse
import math

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

import torchjd
from torchjd.aggregation import UPGrad


class VariationalAutoEncoder(nn.Module):
    def __init__(self, nb_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nb_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels, 2 * latent_dim, kernel_size=4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, nb_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, nb_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, nb_channels, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, nb_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels, 1, kernel_size=5),
        )

    def encode(self, x):
        output = self.encoder(x).view(x.size(0), 2, -1)
        mu, log_var = output[:, 0], output[:, 1]
        return mu, log_var

    def decode(self, z):
        mu = self.decoder(z.view(z.size(0), -1, 1, 1))
        return mu, mu.new_zeros(mu.size())


def sample_gaussian(mu, log_var):
    std = log_var.mul(0.5).exp()
    return torch.randn(mu.size(), device=mu.device) * std + mu


def log_p_gaussian(x, param):
    mean, log_var, x = param[0].flatten(1), param[1].flatten(1), x.flatten(1)
    var = log_var.exp()
    return -0.5 * (((x - mean).pow(2) / var) + log_var + math.log(2 * math.pi)).sum(1)


def dkl_gaussians(param_a, param_b):
    mean_a, log_var_a = param_a[0].flatten(1), param_a[1].flatten(1)
    mean_b, log_var_b = param_b[0].flatten(1), param_b[1].flatten(1)
    var_a = log_var_a.exp()
    var_b = log_var_b.exp()
    return 0.5 * (log_var_b - log_var_a - 1 + (mean_a - mean_b).pow(2) / var_b + var_a / var_b).sum(
        1
    )


def train_model(model, dl, nb_epochs, lr, latent_dim):
    optimizer = SGD(model.parameters(), lr=lr)

    for e in range(nb_epochs):
        for x, _ in dl:
            mu, log_var = model.encode(x)
            z = sample_gaussian(mu, log_var)
            param_p_X_given_z = model.decode(z)
            log_p_x_given_z = log_p_gaussian(x, param_p_X_given_z)

            param_p_Z = (
                torch.zeros((x.size(0), latent_dim), device=x.device),  # Prior mean = 0
                torch.zeros((x.size(0), latent_dim), device=x.device),  # Prior log variance = 0
            )

            dkl_q_Z_given_x_from_p_Z = dkl_gaussians((mu, log_var), param_p_Z)

            reconstruction_loss = -log_p_x_given_z.mean()
            divergence_loss = dkl_q_Z_given_x_from_p_Z.mean()

            loss = (reconstruction_loss + divergence_loss).mean()

            print(loss)

            optimizer.zero_grad()
            torchjd.mtl_backward(
                losses=[reconstruction_loss, divergence_loss],
                features=[mu, log_var],
                tasks_params=[model.decoder.parameters(), []],
                shared_params=model.encoder.parameters(),
                A=UPGrad(),
            )
            optimizer.step()

    return model


def main(nb_epochs, lr, bs, d, n_channels):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    ds = MNIST(root=".", transform=transform, train=True, download=True)
    dl = DataLoader(ds, batch_size=bs)
    model = VariationalAutoEncoder(n_channels, d)
    train_model(model, dl, nb_epochs, lr, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a Variational AutoEncoder on MNIST")

    parser.add_argument("--nb_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--nb_channels", type=int, default=32)

    args = parser.parse_args()

    main(args.nb_epochs, args.learning_rate, args.batch_size, args.latent_dim, args.nb_channels)
