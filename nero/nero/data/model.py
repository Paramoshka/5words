import torch.nn as nn
import torch.nn.functional as F


#https://www.appsilon.com/post/visualize-pytorch-neural-networks
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=4, out_features=16)
        self.hidden_1 = nn.Linear(in_features=16, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=3)

    def forward(self, x):
        x = F.relu_(self.input(x))
        x = F.relu_(self.hidden_1(x))
        return self.output(x)

