from django.core.management import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):

        try:
            lines = open('words.txt').readlines()
            for line in lines[:10]:
                print(line)
        except FileNotFoundError:
            print('words.txt not found')
