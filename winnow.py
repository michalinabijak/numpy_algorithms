import numpy as np
from helper_functions import error


class Winnow:

    def __init__(self):
        self.w = None

    def train(self, train_x, train_y):
        n = train_x.shape[1]
        w = np.ones(n)

        for index, x in enumerate(train_x):
            y_hat = 1 * (x.dot(w) >= n)

            if y_hat != train_y[index]:
                if train_y[index] - y_hat == 1:
                    w *= (2 ** x)
                elif train_y[index] - y_hat == -1:
                    w = w / (2 ** x)

        self.w = w
        pred = 1 * (train_x.dot(self.w) >= n)
        train_err = error(pred, train_y)
        return train_err

    def predict(self, test_x):
        n = test_x.shape[1]
        y_hat = 1 * (test_x.dot(self.w) >= n)
        return y_hat
