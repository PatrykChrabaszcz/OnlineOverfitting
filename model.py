import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch


class RnnBase(nn.Module):
    ModeZeros = 'Zeros'
    ModeRandom = 'Random'

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def initial_state(self, mode):
        if mode == self.ModeZeros:
            return np.zeros((self.num_layers, self.hidden_size), dtype=np.float32)
        elif mode == self.ModeRandom:
            random_state = np.random.normal(0, 1.0, (self.num_layers, self.hidden_size))
            return np.clip(random_state, -1, 1).astype(dtype=np.float32)
        else:
            raise RuntimeError('No mode %s' % mode)

    def initial_states(self, mode, samples=64):
        states = [self.initial_state(mode) for _ in range(samples)]
        states = np.stack(states)
        states = np.swapaxes(states, 1, 0)
        states = Variable(torch.from_numpy(states), requires_grad=False)
        return states


class SimpleRNN(RnnBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden):

        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()

        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        return fc_out, hidden
