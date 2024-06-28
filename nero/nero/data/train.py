import torch
from torch import optim, nn

from torch.utils.data import DataLoader

from nero.data.model import RNN


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    """
    Trains the model for the specified number of epochs
    Inputs
    ------
    model: RNN model to train
    data: Iterable DataLoader
    epochs: Number of epochs to train the model
    optimizer: Optimizer to use for each epoch
    loss_fn: Function to calculate loss
    """

    model.train()

    for epoch in range(epochs):
        for batch in data:
            # skip batch if it doesnt match with the batch_size
            if batch.shape[0] != model.batch_size:
                continue
            hidden = model.init_zero_hidden(batch_size=model.batch_size)
            for c in range(batch.shape[1]):
                #print("Batch: " + str(batch[:, c]) + "\n")
                out = model(batch[:, c].reshape(batch.shape[0], 1), hidden)
                print(out)
                #out, hidden = model(batch[:, c].reshape(batch.shape[0], 1), hidden)

