import numpy as np
import math
import torch


class DatasetBatchIterator:
    "Iterates over labaled dataset in batches"

    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X = self.X[index]
            Y = self.Y[index]

        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] / batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        X_batch = torch.LongTensor(self.X[k * bs : (k + 1) * bs])
        Y_batch = torch.FloatTensor(self.Y[k * bs : (k + 1) * bs])

        return X_batch, Y_batch.view(-1, 1)
