import numpy as np
from scipy.spatial.distance import cdist


def grammatrix(data1, data2, kernel, param):
    """
    :param data1:
    :param data2:
    :param kernel: "p" for polynomial, "g" for gaussian
    :param param: kernel parameter, degree for polynomial, Beta value for gaussian
    :return:
    """
    if kernel == "p":
        gram = data1 @ data2.T
        gram = np.power(gram, param)

    elif kernel == "g":
        data1_sq = np.sum(data1 ** 2, axis=-1, keepdims=True)
        data2_sq = np.sum(data2 ** 2, axis=-1, keepdims=True)
        norms = data1_sq + data2_sq.T - 2 * (data1 @ data2.T)
        gram = np.exp(-param * norms)
    else:
        raise ValueError('Invalid kernel type.')
    return gram


def mysign(x):
    output = np.sign(x)
    output[output == 0] = -1
    return output


def my_sign(x):
    pred = -1 if x <= 0 else 1  # TODO: fix mysign function
    return pred


def classify(x):
    return 0 if x == -1 else 1

def error(y_pred, y):
    err = 1 - np.count_nonzero(y_pred == y) / len(y)
    return err

def get_indices(data_y, alphas):
    predictor_indices = []
    class_indices = [np.where(data_y == i)[0] for i in range(10)]
    for i, alpha in enumerate(alphas):
        indices = np.concatenate((class_indices[alpha[0]], class_indices[alpha[1]]))
        predictor_indices.append(indices)
    return predictor_indices


# helpers for kNN

def get_distance(train_x, test_x, metric):
    dist = cdist(train_x, test_x, metric)
    return dist.flatten()


def get_knn(train_x, test_x, metric, k):
    dist = get_distance(train_x, test_x, metric)
    k_nearest_neighbours = np.argpartition(dist, int(k))[:k]
    return k_nearest_neighbours


# MLP

def ReLU(x):
    return np.maximum(0, x)


def ReLU_prime(values):
    result = 1 * (values >= 0)
    return result


def loss(target, nn_out):
    result = - np.log(nn_out[int(target)])
    return result


def softmax(x):
    expA = np.exp(x - np.max(x))
    return expA / expA.sum(axis=0)


def numericalgradient(fun,w,e):
    """
    Provides a numerical estimate of the gradient of fun w.r.t. to parameters w
    :param e: epsilon
    :return: numerical gradient estimate (shape of w)
    """
    # get dimensionality
    d = len(w)
    # initialize numerical derivative
    dh = np.zeros(d)
    # go through dimensions
    for i in range(d):
        # copy the weight vector
        nw = w.copy()
        # perturb dimension i
        nw[i] += e
        # compute loss
        l1, temp = fun(nw)
        # perturb dimension i again
        nw[i] -= 2*e
        # compute loss
        l2, temp = fun(nw)
        # the gradient is the slope of the loss
        dh[i] = (l1 - l2) / (2*e)
    return dh
