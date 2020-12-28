import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from hyperparams import hyperparams as hp
import matplotlib.pyplot as plt
import sys

cuda = True if torch.cuda.is_available() else False

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.dense_size = 20480
        self.layer = torch.nn.Linear(self.dense_size ,hp.num_classes)

    def forward(self, x):
        plt.imshow(x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3],x.shape[4]).data.numpy()[0].T)
        plt.show()
        input("continue?")
        return self.layer(x.view(x.shape[0], self.dense_size))


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
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=hp.lr)
        self.loss = nn.CrossEntropyLoss()
        self.log = {key:[] for key in ["accuracy", "loss"]}

        if cuda:
            self.model.cuda()
            self.loss.cuda()

    def init_epoch(self):
        self.log["accuracy"].append(0)
        self.log["loss"].append(0)

    def log_entry(self, label):
        self.log["accuracy"][-1] += self.accuracy(self.current_prediction.data.numpy(), label.data.numpy()).data.tolist()
        self.log["loss"][-1] += self.current_loss

    def verbose(self):
        print("epoch: " + str(hp.n_epochs) + " loss: ", str(self.log["loss"][-1].data.tolist()),
              "accuracy: " + str(self.log["accuracy"][-1] / (self.n_batches + 1)))

    def train(self):
        self.init_training()

        for n in range(hp.n_epochs):
            self.init_epoch()

            for i, (data, label) in enumerate(self.dataloader):
                self.step(data, label)
                self.log_entry(label)

            if hp.verbose:
                self.verbose()

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred, 1) == target).sum() / len(y_pred)


if __name__ == "__main__":
    test = TestModule()
    print(test(torch.Tensor(np.random.random((32, 64))).view((1, 1, 32, 64))))
    trainer = Trainer()
    trainer.train()
