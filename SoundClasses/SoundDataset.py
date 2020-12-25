from torch.utils.data import Dataset
import torch
from SoundClasses.SoundFeature import SoundFeature
import numpy as np
import os
from hyperparams import hyperparams as hp
import time
import pandas as pd

class SoundDataset(Dataset):
    def __init__(self):
        self.metadata = {"path": [], "name": [], "shape": [], "duration": [], "n_samples": [], "label": [],
                         "load_time": []}
        self.data, self.labels = self.load()
        assert len(self.data) == len(self.labels), "load error"
        self.n_samples = len(self.data)
        self.process()

    def save_metadata(self, path, name, shape, duration, n_samples, label, timestamp):
        self.metadata["path"].append(path)
        self.metadata["name"].append(name)
        self.metadata["shape"].append(shape)
        self.metadata["duration"].append(duration)
        self.metadata["n_samples"].append(n_samples)
        self.metadata["label"].append(label)
        self.metadata["load_time"].append(timestamp)
        pd.DataFrame(self.metadata).to_csv("files/metadata.csv", sep=",")

    def load(self):
        file_paths = self.get_paths()
        data, labels = [], []
        N = len(file_paths)
        for n, file in enumerate(file_paths):
            if hp.verbose:
                print("file {} von {}: {}".format(n, N, file))

            audio = SoundFeature(hp.path + "/" + file)

            audio.load()
            audio = self.raw_audio_preprocessing(audio)

            feature = audio.get_feature()
            data.append(feature)
            label = self.get_label_info(file)
            labels.append(label)

            self.save_metadata(path=hp.path, name=file, shape=feature.shape, duration=audio.duration,
                               n_samples=audio.n_samples, label=label, timestamp=time.time())
        return data, labels

    def raw_audio_preprocessing(self, audio):
        audio.trim()
        audio.preemphasis()
        audio.stft()
        return audio

    def get_label_info(self, file):   #TODO make this more dynamic.
        alphanumeric_filter = filter(str.isalpha, file[:-4])
        return "".join(alphanumeric_filter)

    def slice_data(self):
        slices = {}
        all_files = os.listdir(hp.path)
        file_paths = all_files[:hp.load_n]
        for n, file in enumerate(file_paths):
            str_label = self.get_label_info(file)
            if str_label not in slices.keys():
                slices[str_label] = []
            slices[str_label].append(file)
        return slices

    def create_training_sets(self):
        slices = self.slice_data()
        file_sets = {key: [] for key in ["train", "test", "valid"]}
        for slice in slices.keys():
            files_from_slice = slices[slice]
            np.random.shuffle(files_from_slice)
            N = len(files_from_slice)
            _train = int(N * hp.split["train"])
            _test = int(N * hp.split["train"]) + int(N * hp.split["test"])
            file_sets["train"].append(files_from_slice[:_train])
            file_sets["test"].append(files_from_slice[_train:_test])
            file_sets["valid"].append(files_from_slice[_test:])

        training_sets = {"file": [], "set": []}
        for set in ["train", "test", "valid"]:
            for files in file_sets[set]:
                for file in files:
                    training_sets["file"].append(file)
                    training_sets["set"].append(set)
        pd.DataFrame(training_sets).to_csv("files/training_sets.csv", sep=",")
        return training_sets


    def get_paths(self, set="all"):
        if set == "all":
            file_paths = os.listdir(hp.path)[:hp.load_n]
        else:
            if os.path.isfile("files/training_sets.csv"):
                training_sets = pd.read_csv("files/training_sets.csv", sep=",")
            else:
                training_sets = self.create_training_sets()
            file_paths = training_sets["file"][training_sets["set"] == set]
        return file_paths

    def process(self):
        self.process_feature()
        self.process_label()

    def process_feature(self):
        max_N = np.max(np.asarray(self.metadata["shape"])[:, 0]) // hp.temporal_rate + 1

        for i, feature in enumerate(self.data):
            # pad for a multiple N of the temporal_rate:
            # (T, n_features) -> (N * temporal_rate, n_features)
            T = feature.shape[0]
            pad = hp.temporal_rate - (T % hp.temporal_rate) if T % hp.temporal_rate != 0 else 0
            feature = np.pad(feature, [[0, pad], [0, 0]], mode="constant")

            # reshape:
            # (N * temporal_rate, n_features) -> (N, temporal_rate, n_features) #TODO need a window function?
            N = int(feature.shape[0] / hp.temporal_rate)
            feature = feature.reshape((N, hp.temporal_rate, hp.n_features))

            # to be able to create proper batches!
            # pad for max N in all data:
            # (N, temporal_rate, n_features) -> (max_N, temporal_rate, n_features)
            pad = max_N - feature.shape[0]
            feature = np.pad(feature, [[0, pad], [0, 0], [0, 0]], mode="constant")

            # reshape for a CNN with 1 channel:
            # (max_N, temporal_rate, n_features) -> (max_N, 1, temporal_rate, n_features)
            # #TODO: implement finalize shape function and use here
            self.data[i] = feature.reshape((max_N, 1, hp.temporal_rate, hp.n_features))

        self.data = torch.from_numpy(np.asarray(self.data) / 172).type(torch.float32)

    def finalize_shape(self):
        # if hp.model_name == "RNN":
        #     self.data = [np.reshape(mel, (hp.temporal_rate, hp.n_features)) for mel in self.data]
        #     self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)
        # elif hp.model_name == "CNN":
        #     self.data = torch.from_numpy(np.asarray(self.data)/172)
        #     self.data = self.data.type(torch.float32)
        # elif hp.model_name == "AudioEncoder":
        #     self.data = [np.reshape(mel, (hp.temporal_rate, hp.n_features)).T for mel in self.data]
        #     self.data = torch.from_numpy(np.asarray(self.data)/172).type(torch.float32)
        pass

    def process_label(self):
        if hp.label_mode == "multiclass":
            unique_labels = np.unique(self.labels)
            self.label2idx = {label: index for index, label in enumerate(unique_labels)}
            self.idx2label = {index: label for index, label in enumerate(unique_labels)}
            self.n_unique_labels = len(np.unique(self.labels))
            self.labels = [self.label2idx[label] for label in self.labels]
        elif hp.label_mode == "binary":
            unique_labels = np.unique(self.labels)
            self.label2idx = {prefix: 1 if prefix == hp.binary_class else 0 for prefix in unique_labels}
            self.idx2label = {1 if prefix == hp.binary_class else 0: prefix for prefix in unique_labels}
            self.n_unique_labels = 2
            self.labels = [1 if prefix == hp.binary_class else 0 for prefix in self.labels]
        if hp.one_hot:
            one_hot_labels = np.zeros((self.n_samples, self.n_unique_labels))
            for i, label in enumerate(self.labels):
                one_hot_labels[i, label] = 1
            self.labels = one_hot_labels
        self.labels = torch.from_numpy(np.asarray(self.labels)).type(torch.long)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    sd = SoundDataset()
    slices = sd.slice_data()
    for key in slices.keys():
        print(key, len(slices[key]))
