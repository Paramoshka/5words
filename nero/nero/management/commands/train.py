from django.core.management import BaseCommand
from nero.data.data import Data
from nero.data.model import Predictor

#https://botfactory.in/2023/06/07/building-a-simple-rnn-neural-network-with-pytorch-a-step-by-step-guide/
class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write("Training neuron", ending="")
        model = Data()
        predictor = Predictor()
