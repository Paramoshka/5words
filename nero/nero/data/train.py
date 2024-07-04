import torch

from nero.data.data import Data
from nero.data.model import RNN


def train(model: RNN, data: Data, epochs: int, alpha) -> None:
    model.train()
    dataset = get_dataset(data)
    for epoch in range(epochs):
        hidden_layer = model.init_hidden()
        for target, input_word in dataset.items():
            for li in range(input_word.size()[0]):
                output, hidden_layer = model(input_word[li], hidden_layer)
                print(output)


def get_dataset(data: Data) -> dict:
    d = dict()
    words = data.five_words
    for w in words[:1]:
        input_tensor = data.word_to_tensor(w)
        target_tensor = data.word_to_target_tensor(w)
        d[target_tensor] = input_tensor
    return d
