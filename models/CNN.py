import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch import optim


class CNN(nn.Module):
    def __init__(self, hp):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 2, 1)
        self.conv3 = nn.Conv2d(256, 346, 2, 1)
        self.conv4 = nn.Conv2d(346, 456, 2, 1)
        self.conv5 = nn.Conv2d(456, 512, 2, 1)
        self.dropout = nn.Dropout(0.3)

        self.dense = nn.Linear(2560, hp.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout(x)
        x = torch.flatten(x, 1)

        x = self.dense(x)
        output = F.softmax(x, dim=1)
        return output


class Trainer:
    def __init__(self, hp, dataloader, model):
        self.hp = hp
        self.model = model
        self.dataloader = dataloader

    def train(self):
        self.model.float()
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.hp.lr)
        loss = nn.BCELoss()
        for n in range(100):
            accuracy = 0
            for i, (data, label) in enumerate(self.dataloader):
                optimizer.zero_grad()
                prediction = self.model(data)
                output = loss(prediction, label)
                output.backward()
                optimizer.step()
                accuracy += self.accuracy(prediction.data, label.data).data.tolist()
            print("epoch: " + str(n) + " loss: ", str(output.data.tolist()),
                    "accuracy: " + str(accuracy/(i+1)))

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred) == np.argmax(target)).sum() / len(y_pred)