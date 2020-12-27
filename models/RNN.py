import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from hyperparams import hyperparams

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        '''Predict one label from one sample's features'''
        # x: feature from a sample, LxN
        #   L is length of sequency
        #   N is feature dimension
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index

    def set_classes(self, classes):
        self.classes = classes

    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        assert self.classes, "Classes names are not set. Don't know what audio label is"
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T  # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx

class Trainer:
    def __init__(self, train_loader, model):
        self.train_loader = train_loader
        self.model = model

    def train(self):
        device = self.hp.device
        # -- Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.lr)
        optimizer.zero_grad()

        # -- Train the model
        total_step = len(self.train_loader)
        curr_lr = self.hp.lr
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

                # Forward pass
                prediction = self.model(featuress)
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
