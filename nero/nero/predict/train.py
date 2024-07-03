import torch
from torch import nn
import time
import math
from nero.predict.model import RNN


def train(model: RNN, data_lines: [], epochs: int, criterion, all_categories, alpha):
    model.train()
    print_every = 100
    start = time.time()

    for epoch in range(epochs):
        loss = torch.Tensor([0])
        hidden_layer = model.init_hidden()
        model.zero_grad()
        for w in data_lines:
            category = w[0]
            word = w[1]
            category_tensor = w[2]
            input_tensor = w[3]

            for i in range(input_tensor.size()[0]):
                output, hidden_layer = model(input_tensor[i], hidden_layer)


                if epoch % print_every == 0:
                    guess, guess_i = category_from_output(output, all_categories)
                    correct = '✓' if guess == category else '✗ (%s)' % category
                    print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / epochs * 100, time_since(start), loss, word, guess, correct))

                loss += criterion(output, category_tensor)
        loss.backward()

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=alpha)



def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)