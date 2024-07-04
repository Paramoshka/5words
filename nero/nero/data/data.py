import torch


def read_lines(filename):
    lines = open(filename, encoding='windows-1251').read().strip().split('\n')
    return lines

def get_5words_list(words: list) -> list:
    five_words = list()
    for word in words:
        if len(word) == 6:
            five_words.append(word)
    return five_words

class Data(object):

    filename = '/home/django/backend/russian.txt'
    def __init__(self):
        self.cyrillic_letters = ''.join(map(chr, range(ord('А'), ord('я') + 1))) + 'Ёё'
        words = read_lines(self.filename)
        self.five_words = get_5words_list(words)
    def get_len_alphabet(self) -> int:
        return len(self.cyrillic_letters)

    def get_alphabet(self) -> list:
        return list(self.cyrillic_letters)

    def letter_to_tensor(self, letter) -> torch.Tensor:
        tensor = torch.zeros(1, len(self.cyrillic_letters))
        index = self.cyrillic_letters.index(letter)
        tensor[0][index] = 1
        return tensor

    def word_to_tensor(self, word) -> torch.Tensor:
        tensor = torch.zeros(len(word), 1, len(self.cyrillic_letters))
        for li, letter in enumerate(word):
            index = self.cyrillic_letters.index(letter)
            tensor[li][0][index] = 1
        return tensor
