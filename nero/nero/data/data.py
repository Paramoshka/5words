import csv
import numpy as np



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

        self.int_arr = np.empty((len(arr), len(arr[0])))

        for k, v in enumerate(arr):
            self.int_arr[k] = v

        mask = np.random.binomial(n=1, p=0.5, size=(len(arr), len(arr[0])))

        self.train_arr = np.multiply(self.int_arr, mask)

    """
    return arrays n x n, labels and train data where random zeros
    [26., 13., 32., 23.,  6.,  0.],
    .....
    [21., 19.,  1., 20.,  1.,  0.]
    """
    def load_data(self):
        return self.int_arr, self.train_arr

