import torch
import torch.nn as nn
import numpy as np
from pkg.modules import MaskedConv1d, HighwayConv1d
from pkg.modules import SequentialMaker
from hyperparams import hyperparams as hp

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        seq = SequentialMaker()
        seq.add_module("conv_0", MaskedConv1d(hp.n_features, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("relu_0", nn.ReLU())
        seq.add_module("drop_0", nn.Dropout(hp.dropout))
        seq.add_module("conv_1", MaskedConv1d(hp.hidden_dim, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("relu_1", nn.ReLU())
        seq.add_module("drop_1", nn.Dropout(hp.dropout))
        seq.add_module("relu_2", MaskedConv1d(hp.hidden_dim, hp.hidden_dim, 1, padding="causal"))
        seq.add_module("drop_2", nn.Dropout(hp.dropout))
        i = 3
        for _ in range(2):
            for j in range(4):
                seq.add_module("highway-conv_{}".format(i),
                               HighwayConv1d(hp.hidden_dim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="causal"))
                seq.add_module("drop_{}".format(i), nn.Dropout(hp.dropout))
                i += 1
        for k in range(2):
            seq.add_module("highway-conv_{}".format(i),
                           HighwayConv1d(hp.hidden_dim,
                                         kernel_size=3,
                                         dilation=3,
                                         padding="causal"))
            if k == 0:
                seq.add_module("drop_{}".format(i), nn.Dropout(hp.dropout))
            i += 1
        seq.add_module("dense", nn.Linear(hp.hidden_dim, hp.num_classes))
        seq.add_module("softmax", nn.Softmax(hp.num_classes))
        self.seq_ = seq()

    def forward(self, inputs):
        return self.seq_(inputs)

    def print_shape(self, input_shape):
        print("audio-encoder {")
        SequentialMaker.print_shape(
            self.seq_,
            torch.FloatTensor(np.zeros(input_shape)),
            intent_size=2)
        print("}")