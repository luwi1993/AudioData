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
        self.conv_block_1 = self.conv_block(1, 16, 2)
        self.conv_block_2 = self.conv_block(16, 32, 2)
        self.conv_block_3 = self.conv_block(32, 64, 2)
        self.attention = Attention(self.dense_size)
        self.hidden_dense = nn.Linear(self.dense_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, hp.num_classes)

    def conv_block(self, in_ch, out_ch, downsample=2):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, [3, 3], [1, 1], [1, 1]),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, [3, 3], [1, 1], [1, 1]),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, [3, 3], [downsample, downsample], [1, 1]),
            nn.ReLU()
        )
        return block

    def conv_embedding(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x

    def forward(self, x):
        batch_size, n_steps, n_channels, hight, width = x.shape
        # querry = torch.cuda.FloatTensor(batch_size, 1, self.dense_size).fill_(0) # use this for gpu
        querry = torch.zeros((batch_size, 1, self.dense_size))
        x = x.view((batch_size * n_steps, n_channels, hight, width))
        x = self.conv_embedding(x)
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
        self.log = {key: [] for key in ["accuracy", "loss"]}

        if cuda:
            self.model.cuda()
            self.loss.cuda()

    def init_epoch(self):
        self.log["accuracy"].append(0)
        self.log["loss"].append(0)

    def log_entry(self, label):
        self.log["accuracy"][-1] += self.accuracy(self.current_prediction.data.numpy(),
                                                  label.data.numpy()).data.tolist()
        self.log["loss"][-1] += self.current_loss

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


if __name__ == "__main__":
    cnn = CNN()
    x = torch.Tensor(np.random.random((32, 64))).view((1, 1, 32, 64))
    x = cnn.conv_block(x, 1, 16, 2)
    x = cnn.conv_block(x, 16, 32, 2)
    x = cnn.conv_block(x, 32, 64, 2)
    cnn(x.view((1, 1, 1, 32, 64)))
