import sys

import torch
from torch import nn

from nero.data.data import Data
from nero.data.model import RNN


def train(model: RNN, data: Data, epochs: int, alpha) -> None:
    model.train()
    dataset = get_dataset(data)
    loss = torch.Tensor([0])
    criterion = nn.NLLLoss()
    hidden_layer = model.init_hidden()
    model.zero_grad()
    print_every = 10

    for epoch in range(epochs):
        for target, input_word in dataset.items():
            for li in range(input_word.size()[0]):
                output, hidden_layer = model(input_word[li], hidden_layer)
                loss += criterion(output, target)
                loss.backward(retain_graph=True)

                for p in model.parameters():
                    p.data.add_(p.grad.data, alpha=-alpha)

                if epoch % print_every == 0:
                    guess, guess_i = word_from_output(output, data)
                    true_word = data.five_words[target.item()]
                    correct = '✓' if guess == true_word else '✗ (%s)' % true_word
                    sys.stdout.write("Gues: " + str(guess) + " -> " + str(guess_i) + " correct: " + str(f'{ correct}') +  '\r')


def get_dataset(data: Data) -> dict:
    d = dict()
    words = data.five_words
    for w in words[:1]:
        input_tensor = data.word_to_tensor(w)
        target_tensor = data.word_to_target_tensor(w)
        d[target_tensor] = input_tensor
    return d


def word_from_output(output, data: Data):
    top_n, top_i = output.topk(1)
    word_i = top_i[0].item()
    return data.five_words[word_i], word_i
