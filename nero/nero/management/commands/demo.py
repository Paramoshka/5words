from django.core.management import BaseCommand
from torch import optim
from torch.utils.data import DataLoader

from nero.demo.data import Data
from nero.demo.model import RNN


class Command(BaseCommand):
    # Hyperparameters
    input_size = 10  # The size of our input sequences
    hidden_size = 16  # The size of our hidden state
    output_size = 10  # The size of our output sequences
    batch_size = 1  # We'll feed batches of size 1 to the network for simplicity
    epochs = 10
    learning_rate = 1e-1
    # Initialize the model and the optimizer
    model = RNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Demo AI'))
        data = DataLoader(Data().sequences, batch_size=self.batch_size)
        for batch in data:
            print(batch)