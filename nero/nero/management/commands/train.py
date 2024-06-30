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
    output_size = 100
    epochs = 5
    loss = nn.CrossEntropyLoss()
    input_size = 6
    # Model
    # 1 because we enter a single number/letter per step.
    rnn_model = RNN(input_size, hidden_size, len(alphabet), output_size)
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.001)

    def handle(self, *args, **options):
        self.stdout.write("Training neuron", ending="")
        train_x, train_y, test_x, test_y = Data().load_data()
        train_loader = DataLoader(train_x, batch_size=self.batch_size)
        train(self.rnn_model, train_loader, 5, self.optimizer, self.loss)
