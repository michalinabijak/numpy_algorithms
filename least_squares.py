import numpy as np
from helper_functions import error, mysign


class LeastSquares:

    def __init__(self):
        self.w = None

    def train(self, train_x, train_y):
        w = np.linalg.pinv(train_x).dot(train_y)
        self.w = w
        y_pred = mysign(np.dot(train_x, self.w))
        train_err = error(y_pred, train_y)

        return train_err

    def predict(self, test_x):
        y_hat = mysign(np.dot(test_x, self.w))
        return y_hat