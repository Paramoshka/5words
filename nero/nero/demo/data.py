import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

class Data(object):

    def __init__(self):
        self.batch_size = 2
        self.sequences = [[i] * 10 for i in range(10)] + [[i + 1] * 10 for i in range(10)]
        self.sequences = [torch.tensor(s, dtype=torch.float) for s in self.sequences]
        self.packed_sequences = nn.utils.rnn.pad_sequence(self.sequences, batch_first=True).view(len(self.sequences), 1, -1)