from django.core.management import BaseCommand
from torch import nn


class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write("Training neuron", ending="")
