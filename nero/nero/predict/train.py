from torch import nn

from nero.predict.model import RNN


def train(model: RNN, data_lines: [], epochs: int, criterion):
    model.train()

    for epoch in range(epochs):
        loss = 0
        hidden_layer = model.init_hidden()
        model.zero_grad()
        for w in data_lines:
            category = w[0]
            word = w[1]
            category_tensor = w[2]
            input_tensor = w[3]

            for i in range(input_tensor.size()[0]):
                output, hidden_layer = model(input_tensor[i], hidden_layer)
                loss += criterion(output, category_tensor)
                print("out: " + str(output))
                print('category_tensor: ' + str(category_tensor))

