from django.core.management.base import BaseCommand

from nero.predict.model import RNN


class Command(BaseCommand):
    hidden_size = 128
    rnn = RNN(n_letters, hidden_size, n_categories)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Predict'))
