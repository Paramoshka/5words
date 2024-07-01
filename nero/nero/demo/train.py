import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nero.demo.model import RNN


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer) -> None:
    for epoch in range(epochs):
        model.train()
        for batch in data:

        # predictions, _ = model(data, None)
        # loss = nn.MSELoss()(predictions.squeeze(), data[:, 1:].squeeze())
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: Loss {loss.item():.4f}")


    # Test the model
    # test_sequence = torch.tensor([0] * 10, dtype=torch.float)
    # prediction, _ = model(test_sequence.unsqueeze(0), None)
    # print(f"Predicted next number: {int(prediction.item())}")