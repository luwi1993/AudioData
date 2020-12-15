class hyperparams:
    def __init__(self):
        self.sample_rate = 22050
        self.nfft = 2048
        self.hop_length = 512
        self.window_length = 2048
        self.eq_mode = "linear"
        self.silence_db = 30
        self.nmels = 80
        self.preemphasis = 0.97
        self.max_db = 100
        self.ref_db = 20