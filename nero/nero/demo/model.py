from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # x is a packed sequence of input tensors
        # h is the initial hidden state
        out, h = self.rnn(x, h)
        # out is a sequence of outputs, we only need the last one
        out = self.output(out[-1])
        return out, h