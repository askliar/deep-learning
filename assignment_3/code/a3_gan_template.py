#!/home/andriis/miniconda3/envs/py36/bin/python

import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ARGS = None
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.init_weights()

    # initialize weigths of the model
    def init_weights(self):
        for module in self.generator.children():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)

    def forward(self, z):
        # Generate images from z
        return self.generator(z.view(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.init_weights()

    # initialize weigths of the model
    def init_weights(self):
        for module in self.discriminator.children():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)

    def forward(self, img):
        # return discriminator score for img
        return self.discriminator(img.view(-1, 1, 64, 64))


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):

    # loss for the discriminator
    loss_D = nn.BCELoss()

    # modified loss for the generator
    loss_G = lambda y: -1 * torch.log(y).mean()

    for epoch in range(ARGS.n_epochs):
        avg_d_loss = 0.0
        avg_g_loss = 0.0
        for _, (imgs, _) in enumerate(dataloader):
            
            # generate labels for real and fake data
            real = torch.FloatTensor([torch.FloatTensor(1, 1).uniform_(0.0, 0.3) if torch.bernoulli(torch.FloatTensor([0.05])) > 0.0
                                    else torch.FloatTensor(1, 1).uniform_(0.7, 1.2) for i in range(imgs.shape[0])]).to(device)
            fake = torch.zeros(imgs.shape[0]).to(device)

            # concatenate real and fake labels
            labels = torch.cat([real, fake]).squeeze().to(device)

            # generate random noise and pass it through the generator to generate images
            z = torch.randn((imgs.shape[0], ARGS.latent_dim)).to(device)
            gen_imgs = generator(z)
            
            imgs = imgs.view(-1, 1, 64, 64).to(device)

            # Train Discriminator
            # -------------------

            # first, train discriminator using previously generated images detached
            # in order to not train the generator
            discriminator.zero_grad()
            optimizer_D.zero_grad()

            # concatenate real and fake images
            data = torch.cat([imgs, gen_imgs.detach()]).view(-1, 1, 64, 64)
            
            discriminator_loss = loss_D(discriminator(data).squeeze(), labels)
            discriminator_loss.backward()
            avg_d_loss += discriminator_loss.item()

            optimizer_D.step()

            # Train Generator
            # ---------------
            # next, train generator using previously generated images
            generator.zero_grad()
            optimizer_G.zero_grad()

            generator_loss = loss_G(discriminator(gen_imgs).squeeze())
            generator_loss.backward()

            avg_g_loss += generator_loss.item()

            optimizer_G.step()

            
        avg_d_loss /= len(dataloader)
        avg_g_loss /= len(dataloader)

        print(
            f"[Epoch {epoch}] | D_loss: {avg_d_loss} | G_loss: {avg_g_loss}")
        # Save Images
        # -----------
        save_image(gen_imgs[:25].detach().view(-1, 1, 64, 64),
                'images/gan/{}.eps'.format(epoch),
                nrow=5, normalize=True)
        # You can save your generator here to re-use it to generate images for your report
        torch.save(generator.state_dict(), os.path.join(
            'models_checkpoints', f'mnist_generator_{epoch}.pt'))


def main():
    # Create output image directory
    os.makedirs('images/gan/', exist_ok=True)
    os.makedirs('models_checkpoints', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(64),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=ARGS.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = nn.DataParallel(Generator(ARGS.latent_dim)).to(device)
    discriminator = nn.DataParallel(Discriminator()).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=ARGS.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=ARGS.lr, betas = (0.5, 0.999))

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    ARGS = parser.parse_args()

    main()
