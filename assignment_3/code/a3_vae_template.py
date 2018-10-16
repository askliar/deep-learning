import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from scipy.stats import norm

import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
        hidden = torch.relu(self.i2h(input))
        mean, logvar = self.h2m(hidden), self.h2s(hidden)
        
        return mean, logvar


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
        hidden = torch.relu(self.z2h(input))
        mean = torch.sigmoid(self.h2m(hidden))

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
        mean, logvar = self.encoder(input)

        eps = torch.randn_like(mean)
        z = eps * torch.exp(0.5 * logvar) + mean

        x_hat = self.decoder(z)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))
        binary_cross_entropy = F.binary_cross_entropy(x_hat, input, reduction='sum')

        # average loss over the batches
        average_negative_elbo = (kl_divergence + binary_cross_entropy)/batch_dim

        return average_negative_elbo

    def sample(self, n_samples=None, z_samples=None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        if n_samples is not None and z_samples is None:
            z_samples = torch.randn(n_samples, self.z_dim).to(device)
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
    os.makedirs(f'images/vae/{ARGS.zdim}', exist_ok=True)

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
        samples, _ = model.sample(n_samples = 25)
        samples = samples.view(25, 1, 28, 28)
        save_image(samples, os.path.join(f'images/vae/{ARGS.zdim}',
            f'{epoch}_{train_elbo}_{val_elbo}_samples.eps'),
                nrow=5, normalize=True
        )

    if ARGS.zdim == 2:
        # Display a 2D manifold of the digits

        # Construct grid of latent variable values
        grid = np.meshgrid(norm.ppf(np.linspace(0.00001, 1.0, 10, endpoint=False)), 
                           norm.ppf(np.linspace(0.00001, 1.0, 10, endpoint=False)))
        cartesian_grid = torch.FloatTensor(np.array(grid).T.reshape((-1, 2))).to(device)
        _, means = model.sample(z_samples = cartesian_grid)
        save_image(means.view(-1, 1, 28, 28), 
                   os.path.join(f'images/vae/{ARGS.zdim}', f'latent.eps'),
            nrow=10, normalize=True
        )
    save_elbo_plot(train_curve, val_curve, f'elbo_{ARGS.zdim}.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
