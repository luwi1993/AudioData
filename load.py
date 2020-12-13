import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class SoundObject:
    def __init__(self, path, nfft = 256):
        self.path = path
        self.nfft = nfft
        self.load()
        self.stft()
        self.istft()

    def __str__(self):
        return "SoundObject:" + "\nsample_rate:"+ str(self.sample_rate) \
                + "\nduration:" + str(self.duration)\
                + "\nn_samples:" + str(self.n_samples)\
                + "\nsamples:" + str(self.samples[:5]) + "..." \
                + "\nfrequenzies:" + str(self.freqs[:3]) + "..." + str(self.freqs[-3:]) \
                + "\ntransform:" + str(self.transform[:3, 1])

    def load(self):
        self.samples, self.sample_rate = librosa.load(self.path)
        self.set_params_from_samples()

    def set_params_from_samples(self):
        self.duration = librosa.get_duration(self.samples)
        self.n_samples = len(self.samples)
        self.time = np.linspace(0, self.duration, self.n_samples)

    def plot_time(self):
        plt.figure()
        plt.plot(self.time[:len(self.samples)], self.samples)
        plt.show()

    def plot_freq(self, mode = "stft"):
        plt.figure()
        if mode == "mean_ft":
            plt.plot(self.freqs[:len(self.transform)], self.transform.mean(axis = 1))
        else:
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(self.transform),ref=np.max),y_axis='log', x_axis='time')
        plt.show()

    def find_beginning(self):
        pass

    def find_end(self):
        pass

    def stft(self):
        self.transform = librosa.stft(self.samples, n_fft= self.nfft)
        self.freqs = librosa.fft_frequencies(self.sample_rate, n_fft= self.nfft)

    def istft(self):
        self.samples = librosa.istft(self.transform)
        self.set_params_from_samples()

    def low_pass(self, upper_bound_freq):
        self.band_pass(self.freqs[0], upper_bound_freq)

    def high_pass(self, lower_bound_freq):
        self.band_pass(lower_bound_freq, self.freqs[-1])

    def band_pass(self, lower_bound_freq, upper_bound_freq, func= self.exp_dec):
        so.stft()
        self.transform[self.freqs <= lower_bound_freq, :] = 0
        self.transform[self.freqs >= upper_bound_freq, :] = 0
        so.istft()

    def exp_dec:
        pass


so = SoundObject("files/voice.wav")

so.plot_freq()
so.plot_time()
print(so)
so.low_pass()
print(so)
so.plot_time()
so.plot_freq()

