import librosa
import librosa.display
import numpy as np
from SoundClasses.SoundIO import SoundIO
from hyperparams import hyperparams as hp



class SoundObject(SoundIO):
    def __init__(self, path):
        super(SoundObject, self).__init__(path)

    def load(self):
        super(SoundObject, self).load()
        self.set_params_from_samples()

    def __str__(self):
        return "\n" + "-" * 100 \
               + "\nSoundObject:" + "\nsample_rate:" + str(self.sample_rate) \
               + "\nduration:" + str(self.duration) \
               + "\nn_samples:" + str(self.n_samples) \
               + "\nsamples:" + str(self.samples[:5]) + "..." \
               + "\nfrequenzies:" + str(self.freqs[:3]) + "..." + str(self.freqs[-3:]) \
               + "\ntransform:" + str(self.transform[:3, 1]) + str(self.transform.shape) \
               + "\n" + "-" * 100

    def get_transforms(self):
        self.stft()
        self.istft()

    def set_params_from_samples(self):
        self.duration = librosa.get_duration(self.samples)
        self.n_samples = len(self.samples)

    def stft(self):
        self.transform = librosa.stft(self.samples, n_fft=hp.nfft, hop_length=hp.hop_length)
        self.freqs = librosa.fft_frequencies(self.sample_rate, n_fft=hp.nfft)

    def istft(self):
        self.samples = librosa.istft(self.transform, hop_length=hp.hop_length)
        self.set_params_from_samples()





import os
import pandas as pd

if __name__ == "__main__":
    res = {"shape0":[], "shape1":[], "min":[], "max":[], "mean":[],"var":[]}
    hpath = hp.path[hp.speakers[0]]
    file_paths = os.listdir(hpath)
    I = len(file_paths)
    for i,file in enumerate(file_paths):
        print(i/I)
        path = hpath + "/" + file
        so = SoundObject(path, init_transforms=True)
        print(so.transform.shape)
    #     feature = so.get_feature()
    #     res["shape0"].append(feature.shape[0])
    #     res["shape1"].append(feature.shape[1])
    #     res["min"].append(feature.min())
    #     res["max"].append(feature.max())
    #     res["mean"].append(feature.mean())
    #     res["var"].append(feature.var())
    # pd.DataFrame(res).to_csv("../files/metadata.csv")