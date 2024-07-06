from django.core.management import BaseCommand
from nero.data.data import Data
from nero.data.model import RNN
from nero.data.train import train


class Command(BaseCommand):
    data = Data()
    hidden_size = 256
    output_size = len(data.five_words)
    input_size = data.get_len_alphabet()
    alpha = 0.01
    epochs = 10
    rnn = RNN(input_size, hidden_size, output_size)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Train model'))

        train(self.rnn, self.data, self.epochs, self.alpha)
