import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from models.Attention import Attention
from hyperparams import hyperparams as hp

cuda = True if torch.cuda.is_available() else False

class CNN(nn.Module):
    def __init__(self, hidden_size=512, dense_size=2048):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.dense_size = dense_size
        self.conv1 = nn.Conv2d(1, 64, [2, 2], [1, 2], [1, 1])
        self.conv2 = nn.Conv2d(64, 128, [2, 2], [1, 1], [1, 1])
        self.conv3 = nn.Conv2d(128, 256, [2, 2], [2, 2], [1, 1])
        self.conv4 = nn.Conv2d(256, 128, [2, 2], [1, 1], [1, 1])
        self.conv5 = nn.Conv2d(128, 64, [2, 2], [1, 2], [1, 1])

        self.dropout = nn.Dropout(0.3)
        self.attention = Attention(self.dense_size)
        self.hidden_dense = nn.Linear(self.dense_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, hp.num_classes)

    def conv_block(self, x, in_ch, out_ch, downsample=2):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, [3, 3], [1, 1], [1, 1]),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, [3, 3], [1, 1], [1, 1]),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, [3, 3], [downsample, downsample], [1, 1]),
            nn.ReLU()
        )
        return block(x)

    def conv_embedding(self, x):
        x = self.conv_block(x, 1, 16, 2)
        x = self.conv_block(x, 16, 32, 2)
        x = self.conv_block(x, 32, 64, 2)
        return x

    def forward(self, input):
        batch_size, n_steps, n_channels, hight, width = input.shape
        querry = torch.zeros((batch_size, 1, self.dense_size))

        x = input.view((batch_size * n_steps, n_channels, hight, width))
        x = self.conv_embedding(x)
        x = torch.flatten(x, 1)
        x = x.view((batch_size, n_steps, self.dense_size))
        querry, attention_weights = self.attention.forward(x, querry)
        embeddings = torch.bmm(attention_weights.view((batch_size, 1, n_steps)), querry)

        embeddings = self.hidden_dense(embeddings.view((batch_size, self.dense_size)))
        embeddings = self.dense(embeddings)
        output = F.softmax(embeddings, dim=1)
        return output


class Trainer:
    def __init__(self, dataloader, model):
        self.model = model
        self.dataloader = dataloader

    def train(self):
        self.model.float()
        optimizer = optim.Adadelta(self.model.parameters(), lr=hp.lr)
        loss = nn.CrossEntropyLoss()

        if cuda:
            self.model.cuda()
            self.model.cuda()
            loss.cuda()

        for n in range(100):
            accuracy = 0
            for i, (data, label) in enumerate(self.dataloader):
                optimizer.zero_grad()
                prediction = self.model(data)

                output = loss(prediction, label)
                output.backward()
                optimizer.step()

                accuracy += self.accuracy(prediction.data.numpy(), label.data.numpy()).data.tolist()

            if hp.verbose:
                print("epoch: " + str(n) + " loss: ", str(output.data.tolist()),
                      "accuracy: " + str(accuracy / (i + 1)))

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred, 1) == target).sum() / len(y_pred)


if __name__ == "__main__":
    cnn = CNN()
    x = torch.Tensor(np.random.random((32, 64))).view((1, 1, 32, 64))
    x = cnn.conv_block(x, 1, 16, 2)
    x = cnn.conv_block(x, 16, 32, 2)
    x = cnn.conv_block(x, 32, 64, 2)
    cnn(x.view((1, 1, 1, 32, 64)))
