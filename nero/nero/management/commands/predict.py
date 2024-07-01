import string
from django.core.management.base import BaseCommand
from nero.predict.model import RNN
from nero.predict.data import Data


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
class Command(BaseCommand):
    hidden_size = 128
    #rnn = RNN(n_letters, hidden_size, n_categories)
    data = Data()
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Predict'))
        print(self.data.category_lines)