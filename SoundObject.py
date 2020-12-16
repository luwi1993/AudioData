import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import animate
import soundfile as sf
#from python_speech_features import mfcc
import sys


class SoundObject:
    def __init__(self, path, hp, init_transforms=True):
        self.path = path
        self.hp = hp
        self.load()
        if init_transforms:
            self.get_transforms()

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

    def load(self):
        self.samples, self.sample_rate = librosa.load(self.path, sr=self.hp.sample_rate)
        assert self.sample_rate == self.hp.sample_rate, "SamplerateError"
        self.set_params_from_samples()

    def set_params_from_samples(self):
        self.duration = librosa.get_duration(self.samples)
        self.n_samples = len(self.samples)
        self.time = np.linspace(0, self.duration, self.n_samples)

    def plot_time(self):
        plt.figure()
        plt.plot(self.time[:len(self.samples)], self.samples)
        plt.show()

    def plot_freq(self, mode="stft"):
        plt.figure()
        if mode == "mean_ft":
            plt.plot(self.freqs[:len(self.transform)], self.transform.mean(axis=1))
        else:
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.transform), ref=np.max), y_axis='log',
                                     x_axis='time')
        plt.show()

    def animate_stft(self):
        animate.animate_stft("stft", self.transform, self.freqs, self.duration)

    def animate_mel(self):
        animate.animate_stft("mel", self.get_mel().T, np.arange(self.hp.nmels), self.duration)

    def find_beginning(self):
        pass

    def find_end(self):
        pass

    def stft(self):
        self.transform = librosa.stft(self.samples, n_fft=self.hp.nfft, hop_length=self.hp.hop_length)
        self.freqs = librosa.fft_frequencies(self.sample_rate, n_fft=self.hp.nfft)

    def istft(self):
        self.samples = librosa.istft(self.transform, hop_length=self.hp.hop_length)
        self.set_params_from_samples()

    def low_pass(self, upper_bound_freq):
        self.band_pass(0, upper_bound_freq)

    def high_pass(self, lower_bound_freq):
        self.band_pass(lower_bound_freq, self.freqs[-1])

    def band_pass(self, lower_bound_freq, upper_bound_freq, epsilon=0.000001):
        so.stft()
        self.transform[self.freqs <= lower_bound_freq - epsilon, :] = (self.transform * self.get_filter_curve()(
            np.abs(self.freqs - lower_bound_freq))[:, None])[self.freqs <= lower_bound_freq - epsilon, :]
        self.transform[self.freqs >= upper_bound_freq + epsilon, :] = (self.transform * self.get_filter_curve()(
            np.abs(upper_bound_freq - self.freqs))[:, None])[self.freqs >= upper_bound_freq + epsilon, :]
        so.istft()

    def linear_decay(self, x, alpha=1):
        return -alpha * np.log(x)

    def exponential_decay(self, x, alpha=1):
        return np.exp(- alpha * np.log(x))

    def get_filter_curve(self):
        if self.hp.eq_mode == "zero":
            return lambda x: 0
        elif self.hp.eq_mode == "linear":
            return self.linear_decay
        elif self.hp.eq_mode == "exponential":
            return self.exponential_decay

    def trim(self):
        self.samples, _ = librosa.effects.trim(self.samples, self.hp.silence_db)
        self.set_params_from_samples()
        self.get_transforms()

    def to_wav(self):
        sf.write("files/processed.wav", self.samples, self.sample_rate)

    def magnitude_spec(self):
        return np.abs(self.transform)

    def db_mel(self):
        return 20 * np.log10(np.maximum(1e-5, self.mel_base()))

    def mel_base(self):
        mel_basis = librosa.filters.mel(self.hp.sample_rate, self.hp.nfft, self.hp.nmels)  # (n_mels, 1+n_fft//2)
        return np.dot(mel_basis, self.magnitude_spec())

    def get_mel(self):
        mel = self.db_mel()
        mel = np.clip((mel - self.hp.ref_db + self.hp.max_db) / self.hp.max_db, 1e-8, 1)
        return mel.T.astype(np.float32)

    def mfcc(self):
        return self.get_mel()
            # mfcc(signal=self.samples, samplerate=self.sample_rate, nfft=self.hp.nfft,
            #                 numcep=self.hp.nmels, nfilt=self.hp.nmels)

    def preemphasis(self, y):
        return np.append(y[0], y[1:] - hp.preemphasis * y[:-1])


from hyperparams import hyperparams
import os
import pandas as pd

if __name__ == "__main__":
    hp = hyperparams()
    res = {"shape0":[], "shape1":[], "min":[], "max":[], "mean":[],"var":[]}
    file_paths = os.listdir(hp.path)
    I = len(file_paths)
    for i,file in enumerate(file_paths):
        print(i/I)
        path = hp.path + "/" + file
        so = SoundObject(path, hp, init_transforms=True)
        mfcc_feature = so.get_mel()
        res["shape0"].append(mfcc_feature.shape[0])
        res["shape1"].append(mfcc_feature.shape[1])
        res["min"].append(mfcc_feature.min())
        res["max"].append(mfcc_feature.max())
        res["mean"].append(mfcc_feature.mean())
        res["var"].append(mfcc_feature.var())
    pd.DataFrame(res).to_csv("files/metadata.csv")