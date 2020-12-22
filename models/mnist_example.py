
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
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


class SoundDataset(Dataset):
    def __init__(self):
        # data loading
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


from mnist import MNIST
import numpy as np

class MnistDataset(Dataset):
    def __init__(self):
        mndata = MNIST('../files/mnist/')
        self.images, self.labels = mndata.load_training()
        assert len(self.images) == len(self.labels), "MNIST Load Error"
        self.n_samples = len(self.images)
        self.prep()

    def prep(self):
        self.prep_images()
        self.prep_labels()

    def prep_images(self):
        self.images = [np.reshape(image, (1, 28, 28)) for image in self.images]
        self.images = torch.from_numpy(np.asarray(self.images)/255).type(torch.float32)


    def prep_labels(self):
        # self.n_unique_labels = len(np.unique(self.labels))
        # self.labels = torch.from_numpy(np.asarray(self.labels)).type(torch.long)    for nllloss

        unique_labels = np.unique(self.labels)
        self.label2idx = {label: index for index, label in enumerate(unique_labels)}
        self.idx2label = {index: label for index, label in enumerate(unique_labels)}
        n_unique_labels = len(unique_labels)
        one_hot_labels = np.zeros((self.n_samples, n_unique_labels))
        for index, label in enumerate(self.labels):
            one_hot_labels[index, self.label2idx[label]] = 1
        self.labels = torch.from_numpy(np.asarray(one_hot_labels)).type(torch.float32)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    lr = 0.00005

    dataset = MnistDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=2)
    torch.save(dataloader, "files/mnist_dataloader.pth")
    # model = CNN()
    # model.float()
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # for i, (data, label) in enumerate(dataloader):
    #     optimizer.zero_grad()
    #     prediction = model(data)
    #     loss =  nn.BCELoss()
    #     output = loss(prediction, label)
    #     output.backward()
    #     optimizer.step()
    #     print("batch_idx: " + str(i) + " loss: ", str(output.data.tolist()))