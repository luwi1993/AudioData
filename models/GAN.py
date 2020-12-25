
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


class Gan:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()



class Trainer:
    def __init__(self, hp, dataloader, model):
        self.gan = model
        self.dataloader = dataloader
        self.img_shape = (1, hp.temporal_rate, hp.n_features)

    def train(self):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        if cuda:
            self.gan.generator.cuda()
            self.gan.discriminator.cuda()
            adversarial_loss.cuda()

        # Configure data loader
        os.makedirs("../../data/mnist", exist_ok=True)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.gan.generator.parameters(), lr=hp.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.gan.discriminator.parameters(), lr=hp.lr, betas=(0.5, 0.999))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        for epoch in range(hp.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                b, n, one,h, w = imgs.shape
                imgs = imgs.view(b * n, 1, hp.temporal_rate, hp.n_features)
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], hp.hidden_dim))))

                # Generate a batch of images
                gen_imgs = self.gan.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(self.gan.discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.gan.discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(self.gan.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                if i == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, hp.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                    )

                batches_done = epoch * len(self.dataloader) + i
                if epoch % 100 == 0 and i == 0:
                    print(imgs.data[0, 0].shape, gen_imgs.data[0, 0].shape)
                    plt.imshow(gen_imgs.data[0, 0])
                    plt.show()
                    plt.imshow(imgs.data[0, 0])
                    plt.show()


