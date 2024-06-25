from django.core.management import BaseCommand
from nero.data.alphabet import alphabet
import numpy as np


class Command(BaseCommand):

    def handle(self, *args, **options):
        self.stdout.write("Unterminated line", ending="")
        words = set()
        try:
            lines = open('russian.txt', 'r', encoding='windows-1251').readlines()
            for line in lines[:1000]:
                if len(line) == 6:
                    words.add(line.lower())
            for w in words[:1]:
                word2index(w)
        except FileNotFoundError:
            print('russian.txt not found')


def word2index(word):
    vector = np.zeros(len(alphabet))
    for index, alpa in enumerate(word):
        print("index: " + str(index) + ", alpa: " + str(alpa))