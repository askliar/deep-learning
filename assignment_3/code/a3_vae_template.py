import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(784, hidden_dim)
        self.h2m = nn.Linear(hidden_dim, z_dim)
        self.h2s = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden = F.relu(self.i2h(input))
        mean, std = self.h2m(hidden), self.h2s(hidden)
        
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.z2h = nn.Linear(z_dim, hidden_dim)
        self.h2m = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        hidden = F.relu(self.z2h(input))
        mean = F.sigmoid(self.h2m(hidden))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        batch_dim = input.shape[0]
        mean, std = self.encoder(input)

        eps = torch.randn_like(mean)
        z = eps * std + mean

        x_hat = self.decoder(z)

        var = std.pow(2)
        kl_divergence = (-0.5 * torch.sum(1 + var.log() - mean.pow(2) - var))

        binary_cross_entropy = F.binary_cross_entropy(
            x_hat, input, size_average=False)
        average_negative_elbo = kl_divergence + binary_cross_entropy
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        z_samples = torch.randn(n_samples, self.z_dim)
        means = self.decoder(z_samples)
        sampled_ims, im_means = torch.bernoulli(means), means

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0.0
    for input in data:
        input = input.view(-1, 784).to(device)
        
        if model.training:
            optimizer.zero_grad()
            elbo = model(input)
            elbo.backward()
            optimizer.step()
        else:
            elbo = model(input)

        average_epoch_elbo += elbo.item()
        average_epoch_elbo /= len(data)
    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data_dir = 'generated_images'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pin_memory = True if torch.cuda.is_available() else False
    data = bmnist(pin_memory=pin_memory)[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        samples, _ = model.sample(25)
        samples = samples.view(25, 1, 28, 28)
        grid = make_grid(samples, nrow=5)
        save_image(grid, 
            os.path.join(data_dir, 
                f'{ARGS.zdim}_{epoch}_{train_elbo}_{val_elbo}_samples.eps')
        )
    # --------------------------------------------------------------------
    #  Add functionality to plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    # if ARGS.zdim == 2:
    #     _, means = model.sample(25)
    #     grid = make_grid(means, nrow=5)
    #     save_image(grid, os.path.join(data_dir, f'{epoch}_{train_elbo}_{val_elbo}_latent.eps'))
    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
