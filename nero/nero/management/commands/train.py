from django.core.management import BaseCommand
from torch import optim, nn

from nero.data.data import Data
from nero.data.model import RNN
from torch.utils.data.dataloader import DataLoader
from nero.data.alphabet import alphabet
from nero.data.train import train

#https://botfactory.in/2023/06/07/building-a-simple-rnn-neural-network-with-pytorch-a-step-by-step-guide/
class Command(BaseCommand):

    batch_size = 100
    hidden_size = 100
    epochs = 5
    loss = nn.CrossEntropyLoss()
    # Model
    # 1 because we enter a single number/letter per step.
    rnn_model = RNN(6, hidden_size, len(alphabet), batch_size)
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.001)

    def handle(self, *args, **options):
        self.stdout.write("Training neuron", ending="")
        train_data, test_data = Data().load_data()
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        train(self.rnn_model, train_loader, 5, self.optimizer, self.loss)
