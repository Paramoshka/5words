import random
import string
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
        category = self.data.all_categories[random.randint(0, self.n_categories - 1)]
        line = self.data.category_lines[category]
        train(self.rnn, category, line, )
       # train(self.rnn, )

    def category_tensor(self):
