from helper_functions import *
from scipy.stats import mode


class KNNClassifier:

    def __init__(self, k):
        self.k = k

    def predict(self, metric, trainset, testset):
        '''
        :param metric: distance metric, compatible with cdist fucntion from scipy
        :return: error rate
        '''
        train_x = trainset[:, 1:]
        train_y = trainset[:, 0]

        test_x = testset[:, 1:]
        test_y = testset[:, 0]
        incorrect = 0

        for index, item in enumerate(test_x):
            neighbours = get_knn(train_x, item.reshape(1, -1), metric, self.k)
            pred = mode(train_y[neighbours])
            if pred[0] != test_y[index]:
                incorrect += 1

        error = incorrect / len(test_y)

        return error
