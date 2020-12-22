class hyperparams:
    sample_rate = 22050
    nfft = 2048
    hop_length = 512
    window_length = 2048
    eq_mode = "linear"
    silence_db = 30
    n_features = 80
    preemphasis = 0.97
    max_db = 100
    ref_db = 20
    lr = 0.00000005
    data_shape = (90,80)
    n_epochs = 200
    batch_size = 32
    load_n = 500
    path = "/Users/luwi/Documents/Datasets/Emotional_Speech/SAVEE/AudioData/JE/"
    model_name = "AudioEncoder"
    dataloader_path = "files/dataloader_"+model_name+".pth"
    num_classes = 7
    device = 'cpu'
    temporal_rate = 4
    feature_mode = "mel"
    verbose = True

    dropout = 0.05
    hidden_dim = 256