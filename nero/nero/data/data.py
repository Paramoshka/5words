import csv
import numpy as np
import torch
from torch import nn


class Data(object):

    def __init__(self):
        data = []
        arr = list()

        with open('words.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(np.array(row['vector']))

        for d in data:
            d = str(d).strip('[]')
            d = str(d).split('.')
            d.pop()
            arr.append(np.array(d))

        int_arr = np.empty((len(arr), len(arr[0])))

        for k, v in enumerate(arr):
            int_arr[k] = v

        self.tensor = torch.tensor(int_arr, dtype=torch.float)

        print(self.tensor[:3])
