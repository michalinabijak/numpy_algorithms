import numpy as np


def logistic_loss(w, xTr, yTr):
    """
    :param w: weight vector
    :param xTr: input matrix
    :param yTr: label vector
    :return: loss, gradient at w
    """
    n, d = xTr.shape
    vector = (1 + np.exp(-yTr * xTr.dot(w)))
    loss = np.sum(np.log(vector))

    grad = np.zeros(d)

    for i in range(n):
        numerator = -yTr[i] * xTr[i] * np.exp(-yTr[i] * np.dot(xTr[i], w))
        denominator = 1 + np.exp(-yTr[i] * np.dot(xTr[i], w))
        grad += numerator / denominator

    return loss, grad


def adagrad(func, w, alpha, maxiter, tol=1e-02):
    """

    :param func: function to minimise
    :param w: initial weight vector
    :param alpha: gradient descent stepsize (scalar)
    :param maxiter: max num of interation
    :param tol: if norm(gradient)<tol, it quits (scalar)
    :return: w - final weight vector, loss history
    """

    eps = 1e-06
    losses = []
    d = w.shape
    G = np.zeros(d)

    for i in range(maxiter):
        loss, grad = func(w)

        if np.linalg.norm(grad) < tol:
            break

        losses.append(loss)

        # get G and update value
        G += grad ** 2
        update = (alpha / (np.sqrt(G + eps))) * grad

        # update w
        w -= update

    return w, losses


def hinge(w, xTr, yTr, lmbda):
    """

    :param w: weight vector
    :param xTr: input matrix
    :param yTr: vector of lables
    :param lmbda: regression constant
    :return: loss, gradient
    """

    n, d = xTr.shape

    vector = np.maximum(1 - yTr * np.dot(xTr, w), 0)
    loss = np.sum(vector) + lmbda * np.linalg.norm(w) ** 2

    grad = np.zeros(d)

    for i in range(n):
        v = yTr[i] * np.dot(xTr[i], w)
        grad += 0 if v > 1 else - yTr[i] * xTr[i]

    grad += lmbda * 2 * w

    return loss, grad


def linclassify(w, xTr):
    """ Make a +1/-1 prediction"""
    return np.sign(xTr @ w)


def false_positive(y_hat, y):
    """

    :param y_hat: predicted lables
    :param y: true labels
    :return: number of false positives (mistakenly selected +1)
    """

    fp = np.logical_and(y_hat > 0, y < 0).sum()
    return fp


def false_negative(y_hat, y):
    """
    :param y_hat: predicted labels
    :param y: true labels
    :return: number of false positives (mistakenly selected +1)
    """

    fp = np.logical_and(y_hat < 0, y > 0).sum()
    return fp