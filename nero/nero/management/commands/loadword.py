from django.core.management import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write("Unterminated line", ending="")
        words = set()
        try:
            lines = open('russian.txt', 'r', encoding='windows-1251').readlines()
            for line in lines[:1000]:
                if len(line) == 6:
                    words.add(line.lower())
            for w in words:
                print(w)
        except FileNotFoundError:
            print('russian.txt not found')
