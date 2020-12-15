from torch.utils.data import Dataset, DataLoader
import torch
from SoundObject import SoundObject
from hyperparams import hyperparams
import numpy as np
import os


def load(path):
    hp = hyperparams()
    file_paths = os.listdir(path)[:5]
    mels, labels = [], []
    for file in file_paths:
        audio = SoundObject(path + "/" + file, hp)
        mels.append(audio.get_mel())
        alphanumeric_filter = filter(str.isalpha, file[:-4])
        prefix = "".join(alphanumeric_filter)
        labels.append(prefix)
    return mels, labels


class SoundDataset(Dataset):
    def __init__(self, path):
        self.mels, self.labels = load(path)
        assert len(self.mels) == len(self.labels), "load error"
        self.n_samples = len(self.mels)

    def prep(self):
        self.prep_mels()
        self.prep_labels()

    def prep_mels(self):
        self.mels = [np.reshape(image, (1, 28, 28)) for image in self.images]
        self.mels = torch.from_numpy(np.asarray(self.images)).type(torch.float32)

    def prep_labels(self):
        unique_labels = np.unique(self.labels)
        self.label2idx = {label: index for index, label in enumerate(unique_labels)}
        self.idx2label = {index: label for index, label in enumerate(unique_labels)}
        n_unique_labels = len(unique_labels)
        one_hot_labels = np.zeros((self.n_samples, n_unique_labels))
        for index, label in enumerate(self.labels):
            one_hot_labels[index, self.label2idx[label]] = 1
        self.labels = torch.from_numpy(np.asarray(one_hot_labels)).type(torch.float32)

    def __getitem__(self, index):
        return self.mels[index], self.labels[index]

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    path = "/Users/luwi/Documents/Datasets/Emotional_Speech/SAVEE/AudioData/JE/"
    SoundDataset(path)