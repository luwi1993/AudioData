from torch.utils.data import Dataset
import torch
from SoundClasses.SoundFeatures import SoundFeatures
import numpy as np
import os
from hyperparams import hyperparams as hp


class SoundDataset(Dataset):
    def __init__(self):
        self.data, self.labels = self.load()
        assert len(self.data) == len(self.labels), "load error"
        self.n_samples = len(self.data)
        self.prep()

    def load(self):
        file_paths = os.listdir(hp.path)[:hp.load_n]
        data, labels = [], []
        N = len(file_paths)
        for n,file in enumerate(file_paths):
            if hp.verbose:
                print("file {} von {} : {}".format(n,N, file))
            audio = SoundFeatures(hp.path + "/" + file)
            data.append(audio.get_feature()[:hp.data_shape[0], :])
            alphanumeric_filter = filter(str.isalpha, file[:-4])
            prefix = "".join(alphanumeric_filter)
            labels.append(prefix)
        return data, labels

    def prep(self):
        self.prep_mels()
        self.prep_labels()

    def prep_mels(self):
        if hp.model_name == "RNN":
            self.data = [np.reshape(mel, (hp.data_shape[0], hp.data_shape[1])) for mel in self.data]
            self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)
        elif hp.model_name == "CNN":
            self.data = [np.reshape(mel, (1, hp.data_shape[0], hp.data_shape[1])) for mel in self.data]
            self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)
        elif hp.model_name == "AudioEncoder":
            self.data = [np.reshape(mel, (hp.data_shape[0], hp.data_shape[1])).T for mel in self.data]
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