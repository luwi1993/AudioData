from torch.utils.data import Dataset
import torch
from SoundObject import SoundObject
import numpy as np
import os


def load(hp):
    file_paths = os.listdir(hp.path)[:hp.load_n]
    mels, labels = [], []
    for file in file_paths:
        audio = SoundObject(hp.path + "/" + file, hp)
        mels.append(audio.mfcc()[:hp.data_shape[0],:])
        alphanumeric_filter = filter(str.isalpha, file[:-4])
        prefix = "".join(alphanumeric_filter)
        labels.append(prefix)
    return mels, labels


class SoundDataset(Dataset):
    def __init__(self, hp):
        self.hp = hp
        self.mels, self.labels = load(self.hp)
        assert len(self.mels) == len(self.labels), "load error"
        self.n_samples = len(self.mels)
        self.prep()

    def prep(self):
        self.prep_mels()
        self.prep_labels()

    def prep_mels(self):
        self.mels = [np.reshape(mel, (1, self.hp.data_shape[0], self.hp.data_shape[1])) for mel in self.mels]
        self.mels = torch.from_numpy(np.asarray(self.mels)).type(torch.float32)

    def prep_labels(self):
        unique_labels = np.unique(self.labels)
        self.label2idx = {label: index for index, label in enumerate(unique_labels)}
        self.idx2label = {index: label for index, label in enumerate(unique_labels)}
        self.n_unique_labels = len(unique_labels)
        one_hot_labels = np.zeros((self.n_samples, self.n_unique_labels))
        for index, label in enumerate(self.labels):
            one_hot_labels[index, self.label2idx[label]] = 1
        self.labels = torch.from_numpy(np.asarray(one_hot_labels)).type(torch.float32)

    def __getitem__(self, index):
        return self.mels[index], self.labels[index]

    def __len__(self):
        return self.n_samples

from hyperparams import hyperparams
if __name__ == "__main__":
    hp = hyperparams()
    SoundDataset(hp)