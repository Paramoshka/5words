import glob
import os
import string
import unicodedata

import torch


class Data(object):
    path_names_folder = '/home/django/backend/names/*.txt'
    category_lines = {}
    all_categories = []

    def __init__(self):
        print('init data')
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)

        for filename in glob.glob(self.path_names_folder):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            self.category_lines[category] = lines

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    def letter_to_index(self, letter):
        return self.all_letters.find(letter)

    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letter_to_index(letter)] = 1
        return tensor
