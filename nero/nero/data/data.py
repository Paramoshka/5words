import csv
import numpy as np
class  Data(object):

    def __init__(self):
        self.data = np.array()
        with open('words.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row['vector'])