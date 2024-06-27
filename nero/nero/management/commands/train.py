import torch
from django.core.management import BaseCommand
from nero.data.data import Data
from nero.data.model import Predictor
from torch.utils.data.dataloader import DataLoader
from nero.data.alphabet import alphabet

#https://botfactory.in/2023/06/07/building-a-simple-rnn-neural-network-with-pytorch-a-step-by-step-guide/
class Command(BaseCommand):

    batch_size = 64
    hidden_size = 256
    def handle(self, *args, **options):
        self.stdout.write("Training neuron", ending="")
        train_data, test_data = Data().load_data()
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        # Model
        rnnModel = Predictor(1, self.hidden_size, len(alphabet))  # 1 because we enter a single number/letter per step.

