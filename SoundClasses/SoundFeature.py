from SoundClasses.SoundProcessing import SoundProcessor
import numpy as np
from hyperparams import hyperparams as hp
import librosa

class SoundFeature(SoundProcessor):
    def __init__(self, path):
        super(SoundFeature, self).__init__(path)

    def spectrogram(self):
        return self.transform

    def magnitude_spectrogram(self):
        return np.absolute(self.transform)

    def mel_spectrogram(self):
        mel_basis = self.mel_base()
        return np.dot(mel_basis, self.magnitude_spectrogram())

    def mel_base(self):
        return librosa.filters.mel(hp.sample_rate, hp.nfft, hp.n_features)  # (n_mels, 1+n_fft//2)

    def mfcc(self):
        return librosa.feature.mfcc(self.samples, self.sample_rate, n_mfcc=hp.n_features)

    def temporal_reduction(self):
        self.samples = self.samples[..., ::hp.temporal_rate]
        return self.samples

    def to_db(self, x):
        return 20 * np.log10(np.maximum(1e-10, x))

    def mel_feature(self):
        mel = self.mel_spectrogram()
        mel = self.to_db(mel)
        mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
       # mel = self.pad(mel)
        return mel.T.astype(np.float32)

    def mfcc_feature(self):
        mfcc =  self.mfcc()
        mfcc = self.to_db(mfcc)
        mfcc = np.clip((mfcc - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
        return mfcc.T.astype(np.float32)

    def preemphasis(self):
        return np.append(self.samples[0], self.samples[1:] - hp.preemphasis * self.samples[:-1])

    def get_feature(self):
        if hp.feature_mode == "mel":
            feature = self.mel_feature()
        elif hp.feature_mode == "mfcc":
            feature = self.mfcc_feature()
        return feature

import os
if __name__ == "__main__":
    file_paths = os.listdir(hp.path)[:10]
    for i,file in enumerate(file_paths):
        path = hp.path + "/" + file
        sf = SoundFeature(path)
        feature = sf.get_feature()
        feature = sf.pad(feature)
        n = int(feature.shape[0]/hp.temporal_rate)
        feature = feature.reshape((n, 32, 80))
        print(feature.shape)
