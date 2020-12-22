import librosa
import soundfile as sf
from hyperparams import hyperparams as hp

class SoundIO:
    def __init__(self, path):
        self.path = path

    def load(self):
        self.samples, self.sample_rate = librosa.load(self.path, sr=hp.sample_rate)
        assert self.sample_rate == hp.sample_rate, "SamplerateError"

    def to_wav(self, path):
        sf.write("files/processed.wav", self.samples, self.sample_rate)