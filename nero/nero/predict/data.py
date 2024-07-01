import glob
import os
import string
import unicodedata


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
