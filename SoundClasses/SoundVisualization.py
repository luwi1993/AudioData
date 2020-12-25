import librosa
import matplotlib.pyplot as plt
import numpy as np
from SoundClasses.visualization_utils import animate
from SoundClasses.SoundObject import SoundObject
from hyperparams import hyperparams as hp

class SoundVisualizer(SoundObject):
    def __init__(self, path):
        super(SoundVisualizer, self).__init__(path, init_transforms=True)

    def plot_time(self):
        time = np.linspace(0, self.duration, self.n_samples)
        plt.figure()
        plt.plot(time[:len(self.samples)], self.samples)
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
        animate.animate_stft("mel", self.get_mel().T, np.arange(hp.n_features), self.duration)
