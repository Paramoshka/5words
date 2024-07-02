import random
import string

import torch
from django.core.management.base import BaseCommand
from nero.predict.model import RNN
from nero.predict.data import Data
from nero.predict.train import train

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


class Command(BaseCommand):
    data = Data()
    n_categories = len(data.all_categories)
    hidden_size = 128
    rnn = RNN(n_letters, hidden_size, n_categories)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Predict'))
        category_name = self.data.all_categories[random.randint(0, self.n_categories - 1)]
        line = self.data.category_lines[category_name]
        #train(self.rnn, category_name, line, )
        self.category_tensor(category_name)

    def category_tensor(self, category_name) -> torch.Tensor:
        li = self.data.all_categories.index(category_name)
        tensor = torch.zeros(1, len(self.data.all_categories))
        tensor[0][li] = 1
        return tensor
