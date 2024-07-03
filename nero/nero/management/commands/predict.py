import random
import string

import torch
from django.core.management.base import BaseCommand
from torch import nn, Tensor

from nero.predict.model import RNN
from nero.predict.data import Data
from nero.predict.train import train

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)





class Command(BaseCommand):
    data = Data()
    n_categories = len(data.all_categories)
    hidden_size = 128
    learning_rate = 0.001
    criterion = nn.NLLLoss()
    epochs = 300
    count_samples = 1000
    rnn = RNN(n_letters, hidden_size, n_categories)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Predict'))
        list_data = self.train_data(self.count_samples)
        train(self.rnn, list_data, self.epochs, self.criterion, self.data.all_categories, self.learning_rate)


    def random_choice(l):
        return l[random.randint(0, len(l) - 1)]

    """
    Return n count categories and words
    category, line, category_tensor, line_tensor
    """

    def train_data(self, count_samples):
        sum_data = []
        for _ in range(count_samples):
            category_name = random.choice(self.data.all_categories)
            line = random.choice(self.data.category_lines[category_name])
            category_tensor = torch.tensor([self.data.all_categories.index(category_name)], dtype=torch.long)
            input_line_tensor = self.line_to_tensor(line)
            data = [category_name, line, category_tensor, input_line_tensor]
            sum_data.append(data)

        return sum_data

    def line_to_tensor(self, line) -> torch.Tensor:
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letter_to_index(letter)] = 1
        return tensor

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, n_letters)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor

    # Find letter index from all_letters, e.g. "a" = 0
    def letter_to_index(self, letter):
        return all_letters.find(letter)