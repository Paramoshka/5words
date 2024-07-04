from django.core.management import BaseCommand
from torch import optim, nn

from nero.data.data import Data
from nero.data.model import RNN
from torch.utils.data.dataloader import DataLoader
from nero.data.alphabet import alphabet
from nero.data.train import train

#https://botfactory.in/2023/06/07/building-a-simple-rnn-neural-network-with-pytorch-a-step-by-step-guide/
class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Train model'))

        data = Data()
        print(data.letter_to_tensor('х'))
