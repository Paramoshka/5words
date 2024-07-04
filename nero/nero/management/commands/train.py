from django.core.management import BaseCommand
from torch import optim, nn
from nero.data.data import Data
from nero.data.model import RNN
from nero.data.train import train


class Command(BaseCommand):
    hidden_size = 64
    output_size = 1
    alpha = 0.001
    epochs = 1
    data = Data()
    rnn = RNN(data.get_len_alphabet(), hidden_size, output_size)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Train model'))

        train(self.rnn, self.data, self.epochs, self.alpha)
