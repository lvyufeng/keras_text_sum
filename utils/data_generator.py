from keras.utils import Sequence
import numpy as np
import pandas as pd
import math


class DataGenerator(Sequence):

    def __init__(self,datas, batch_size = 1, shuffle = True):
        # pass
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle


    def __len__(self):
        return math.ceil(len(self.datas)/float(self.batch_size))

    def __getitem__(self, item):
        # generate each batch data


        pass

    def on_epoch_end(self):
        # make a shuffle after a epoch end
        if self.shuffle:
            np.random.shuffle(self.indexes)

        pass