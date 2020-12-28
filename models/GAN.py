
import os
import matplotlib.pyplot as plt 
from torch.autograd import Variable
from hyperparams import hyperparams as hp
import torch
import torch.nn as nn
import numpy as np

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_shape = (1, hp.temporal_rate, hp.n_features)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(hp.hidden_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.img_shape = (1, hp.temporal_rate, hp.n_features)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Gan(nn.Module):
    def __init__(self):
        super(Gan, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()



class Trainer:
    def __init__(self, dataloader, model):
        self.gan = model
        self.dataloader = dataloader
        self.img_shape = (1, hp.temporal_rate, hp.n_features)
        self.n_batches = len(self.dataloader)

    def step(self, imgs, label):
        b, n, one, h, w = imgs.shape
        imgs = imgs.view(b * n, 1, hp.temporal_rate, hp.n_features)
        # Adversarial ground truths
        valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(self.Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], hp.hidden_dim))))

        # Generate a batch of images
        self.current_gen_imgs = self.gan.generator(z)

        # Loss measures generator's ability to fool the discriminator
        self.current_g_loss = self.adversarial_loss(self.gan.discriminator(self.current_gen_imgs), valid)

        self.current_g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.gan.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.gan.discriminator(self.current_gen_imgs.detach()), fake)
        self.current_d_loss = (real_loss + fake_loss) / 2

        self.current_d_loss.backward()
        self.optimizer_D.step()

    def init_training(self):
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        if cuda:
            self.gan.generator.cuda()
            self.gan.discriminator.cuda()
            self. adversarial_loss.cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.gan.generator.parameters(), lr=hp.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.gan.discriminator.parameters(), lr=hp.lr, betas=(0.5, 0.999))

        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.log = {key: [] for key in ["epoch", "d_loss", "g_loss"]}


    def init_epoch(self, epoch):
        self.log["epoch"].append(epoch)
        self.log["d_loss"].append(0)
        self.log["g_loss"].append(0)

    def log_entry(self, label):
        self.log["d_loss"][-1] += self.current_d_loss
        self.log["g_loss"][-1] += self.current_g_loss

    def verbose(self, epoch):
        print(
            "epoch: {}; n_batches: {}; d_loss: {}; g_loss: {}".format(
                epoch,
                self.n_batches,
                self.log["d_loss"][-1],
                self.log["g_loss"][-1])
        )
        if epoch % 1000 == 0:
            plt.imshow(self.current_gen_imgs.data[0, 0].cpu())
            plt.savefig("files/GAN_{}.png".format(epoch))


    def train(self):
        self.init_trianing()
        for epoch in range(hp.n_epochs):
            self.init_epoch(epoch)
            for i, (imgs, _) in enumerate(self.dataloader):
                self.step(imgs, _)

            if hp.verbose:
                self.verbose()



