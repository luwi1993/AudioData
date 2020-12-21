import torch
import torch.nn as nn
import numpy as np
from pkg.modules import MaskedConv1d, HighwayConv1d
from pkg.modules import SequentialMaker
from hyperparams import hyperparams as hp

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        seq = SequentialMaker()
        seq.add_module("conv_0", MaskedConv1d(hp.n_features, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("relu_0", nn.ReLU())
        seq.add_module("drop_0", nn.Dropout(hp.dropout))
        seq.add_module("conv_1", MaskedConv1d(hp.hidden_dim, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("relu_1", nn.ReLU())
        seq.add_module("drop_1", nn.Dropout(hp.dropout))
        seq.add_module("relu_2", MaskedConv1d(hp.hidden_dim, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("drop_2", nn.Dropout(hp.dropout))
        i = 3
        for _ in range(2):
            for j in range(4):
                seq.add_module("highway-conv_{}".format(i),
                               HighwayConv1d(hp.hidden_dim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="causal"))
                seq.add_module("drop_{}".format(i), nn.Dropout(hp.dropout))
                i += 1
        for k in range(2):
            seq.add_module("highway-conv_{}".format(i),
                           HighwayConv1d(hp.hidden_dim,
                                         kernel_size=3,
                                         dilation=3,
                                         padding="causal"))
            if k == 0:
                seq.add_module("drop_{}".format(i), nn.Dropout(hp.dropout))
            i += 1

        seq.add_module("dense", nn.Linear(hp.hidden_dim, hp.num_classes))
        seq.add_module("softmax", nn.Softmax(1))

        self.seq_ = seq()
        self.dense = nn.Linear(hp.hidden_dim, hp.num_classes)

    def forward(self, inputs):
        return nn.Softmax(self.dense(self.seq_(inputs).view(hp.batch_size, hp.hidden_dim, )),1)

    def print_shape(self, input_shape):
        print("audio-encoder {")
        SequentialMaker.print_shape(
            self.seq_,
            torch.FloatTensor(np.zeros(input_shape)),
            intent_size=2)
        print("}")

class Trainer:
    def __init__(self, hp, train_loader, model):
        self.hp = hp
        self.train_loader = train_loader
        self.model = model

    def train(self):
        device = self.hp.device
        # -- Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.lr)
        optimizer.zero_grad()

        # -- Train the model
        cnt_batches = 0
        for n in range(1, 1 + self.hp.n_epochs):
            accuracy = 0
            for i, (featuress, labels) in enumerate(self.train_loader):
                cnt_batches += 1

                ''' original code of pytorch-tutorial:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                # we can see that the shape of images should be: 
                #    (batch_size, sequence_length, input_size)
                '''
                featuress = featuress.to(device)
                labels = labels.to(device)
                print(featuress.shape)
                # Forward pass
                prediction = self.model(featuress)
                print(prediction.shape)
                loss = criterion(prediction, labels)

                # Backward and optimize
                loss.backward()  # error
                optimizer.step()
                optimizer.zero_grad()

                accuracy += self.accuracy(prediction.data, labels.data).data.tolist()

            if self.hp.verbose:
                print("epoch: " + str(n) + " loss: ", str(loss.data.tolist()),
                      "accuracy: " + str(accuracy / (i + 1)))

    def accuracy(self, y_pred, target):
        return (np.argmax(y_pred) == np.argmax(target)).sum() / len(y_pred)