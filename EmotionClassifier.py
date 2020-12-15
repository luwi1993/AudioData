import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from SoundDataset import SoundDataset
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
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
        for n in range(100):
            accuracy = 0
            for i, (data, label) in enumerate(self.dataloader):
                optimizer.zero_grad()
                prediction = self.model(data)

                loss = nn.BCELoss()

                output = loss(prediction, label)
                output.backward()
                optimizer.step()
                accuracy += self.accuracy(prediction.data, label.data).data.tolist()
            print("epoch: " + str(n) + " loss: ", str(output.data.tolist()),
                    "accuracy: " + str(accuracy/(i+1)))

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred) == np.argmax(target)).sum() / len(y_pred)

import os
def dataloader_exists(path):
    return os.path.isfile(path)

from hyperparams import hyperparams
if __name__ == "__main__":
    hp = hyperparams()

    print("prepare dataset...")
    if dataloader_exists(hp.dataloader_path):
        dataloader = torch.load(hp.dataloader_path)
        print("dataloader found at: "+ hp.dataloader_path)
    else:
        dataset = SoundDataset(hp)
        dataloader = DataLoader(dataset=dataset, batch_size=hp.batch_size, shuffle=True, num_workers=0)
        torch.save(dataloader, hp.dataloader_path)

    model = CNN()
    print("start training...")
    trainer = Trainer(hp, dataloader, model)
    trainer.train()
