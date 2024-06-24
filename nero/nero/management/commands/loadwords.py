from django.core.management import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        with open('words.txt', 'r') as f:
