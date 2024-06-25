from django.core.management import BaseCommand
from nero.data.alphabet import alphabet
import numpy as np
import csv


class Command(BaseCommand):

    def handle(self, *args, **options):
        self.stdout.write("load from file", ending="")
        words = set()
        try:
            lines = open('russian.txt', 'r', encoding='windows-1251').readlines()
            for line in lines:
                if len(line) == 6:
                    words.add(line.lower())
            for w in words:
                vector = word2index(w)
                write_word_to_csv(w, vector)
        except FileNotFoundError:
            print('russian.txt not found')


def word2index(word):
    vector = np.zeros(len(word))
    for index, alpa in enumerate(word):
        vector[index] = alphabet[alpa]
    print("word: " + word + " vector: " + str(vector))

    return vector


def write_word_to_csv(word, vector):
    with open('words.csv', 'a', newline='') as csvfile:
        fieldnames = ['word', 'vector']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       # writer.writeheader()
        writer.writerow({'word': word, 'vector': vector})