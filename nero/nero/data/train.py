from torch import optim, nn
from torch.nn import RNN
from torch.utils.data import DataLoader


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
