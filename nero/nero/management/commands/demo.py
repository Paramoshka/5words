from django.core.management import BaseCommand
from torch import optim, nn
from torch.utils.data import DataLoader

from nero.demo.data import Data
from nero.demo.model import RNN
from nero.demo.train import train


class Command(BaseCommand):
    # Hyperparameters
    input_size = 10  # The size of our input sequences
    hidden_size = 16  # The size of our hidden state
    output_size = 10  # The size of our output sequences
    batch_size = 10  # We'll feed batches of size 1 to the network for simplicity
    epochs = 1
    learning_rate = 1e-1

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Demo AI'))

        # Initialize the model and the optimizer

        data = DataLoader(Data().sequences, batch_size=self.batch_size)
        model = RNN(self.input_size, self.hidden_size, self.output_size)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        train(model, data, self.epochs, optimizer)
