import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nero.demo.model import RNN


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    for epoch in range(epochs):
        predictions, _ = model(data, None)
        loss = nn.MSELoss()(predictions.squeeze(), data[:, 1:].squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
