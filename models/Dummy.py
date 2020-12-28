import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from models.Attention import Attention
from hyperparams import hyperparams as hp

cuda = True if torch.cuda.is_available() else False


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        pass


class Trainer:
    def __init__(self, dataloader, model):
        self.model = model
        self.dataloader = dataloader
        self.n_batches = len(self.dataloader)

    def step(self, data, label):
        self.optimizer.zero_grad()
        self.current_prediction = self.model(data)
        self.current_loss = self.loss(self.current_prediction, label)
        self.current_loss.backward()
        self.optimizer.step()

    def init_training(self):
        self.model.float()
        self.optimizer = ""
        self.loss = ""
        self.log = {key: [] for key in ["epoch", "accuracy", "loss"]}

        if cuda:
            self.model.cuda()
            self.loss.cuda()

    def init_epoch(self, epoch):
        self.log["epoch"].append(epoch)
        self.log["accuracy"].append(0)
        self.log["loss"].append(0)

    def log_entry(self, label):
        self.log["loss"][-1] += self.current_loss
        self.log["accuracy"][-1] += self.accuracy(
                                        self.current_prediction.data.numpy(),
                                        label.data.numpy()
                                    ).data.tolist()

    def verbose(self, epoch):
        print(
            "epoch: {}; n_batches: {}; loss: {}; accuracy {}".format(
                epoch,
                self.n_batches,
                self.log["loss"][-1],
                self.log["accuracy"][-1] / (self.n_batches + 1))
        )

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred, 1) == target).sum() / len(y_pred)


