from django.core.management import BaseCommand
from torch import optim, nn
from nero.data.data import Data
from nero.data.model import RNN
from nero.data.train import train

#https://botfactory.in/2023/06/07/building-a-simple-rnn-neural-network-with-pytorch-a-step-by-step-guide/
class Command(BaseCommand):
    hidden_size = 64
    output_size = 1
    data = Data()
    rnn = RNN(data.get_len_alphabet(), hidden_size, output_size)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Train model'))


        test_data = data.five_words[:5]
        train()


