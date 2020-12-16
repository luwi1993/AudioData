from torch.utils.data import Dataset
import torch
from SoundObject import SoundObject
import numpy as np
import os




class SoundDataset(Dataset):
    def __init__(self, hp):
        self.hp = hp
        self.data, self.labels = self.load()
        assert len(self.data) == len(self.labels), "load error"
        self.n_samples = len(self.data)
        self.prep()

    def load(self):
        file_paths = os.listdir(self.hp.path)[:self.hp.load_n]
        data, labels = [], []
        for file in file_paths:
            audio = SoundObject(self.hp.path + "/" + file, self.hp)
            data.append(audio.mfcc()[:self.hp.data_shape[0], :])
            alphanumeric_filter = filter(str.isalpha, file[:-4])
            prefix = "".join(alphanumeric_filter)
            labels.append(prefix)
        return data, labels

    def prep(self):
        self.prep_mels()
        self.prep_labels()

    def prep_mels(self):
        if self.hp.model_name == "RNN":
            self.data = [np.reshape(mel, (self.hp.data_shape[0], self.hp.data_shape[1])) for mel in self.data]
            self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)
        elif self.hp.model_name == "CNN":
            self.data = [np.reshape(mel, (1, self.hp.data_shape[0], self.hp.data_shape[1])) for mel in self.data]
            self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)

    def prep_labels(self):
        unique_labels = np.unique(self.labels)
        self.label2idx = {label: index for index, label in enumerate(unique_labels)}
        self.idx2label = {index: label for index, label in enumerate(unique_labels)}
        self.n_unique_labels = len(np.unique(self.labels))
        self.labels = [self.label2idx[label] for label in self.labels]
        self.labels = torch.from_numpy(np.asarray(self.labels)).type(torch.long)
        # unique_labels = np.unique(self.labels)
        # self.label2idx = {label: index for index, label in enumerate(unique_labels)}
        # self.idx2label = {index: label for index, label in enumerate(unique_labels)}
        # self.n_unique_labels = len(unique_labels)
        # one_hot_labels = np.zeros((self.n_samples, self.n_unique_labels))
        # for index, label in enumerate(self.labels):
        #     one_hot_labels[index, self.label2idx[label]] = 1
        # self.labels = torch.from_numpy(np.asarray(one_hot_labels)).type(torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples

from hyperparams import hyperparams
if __name__ == "__main__":
    hp = hyperparams()
    SoundDataset(hp)