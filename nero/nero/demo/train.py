import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nero.demo.model import RNN


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer) -> None:
    for epoch in range(epochs):
        model.train()
        loss = 0
        for batch in data:
            hidden = model.init_zero_hidden(batch_size=batch.shape[0])

            for tensor_num in range(batch.shape[0]):
                out, hidden = model(batch[tensor_num], hidden)
                loss += nn.MSELoss()(out, batch[tensor_num][0])
                loss.backward(retain_graph=True)
                # optimizer.step()
                # optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")


    # Test the model
    # test_sequence = torch.tensor([0] * 10, dtype=torch.float)
    # prediction, _ = model(test_sequence.unsqueeze(0), None)
    # print(f"Predicted next number: {int(prediction.item())}")