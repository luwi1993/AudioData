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
        self.lr = 0.0000000005
        self.data_shape = (50,80)
        self.batch_size = 10
        self.load_n = 100
        self.path = "/Users/luwi/Documents/Datasets/Emotional_Speech/SAVEE/AudioData/JE/"