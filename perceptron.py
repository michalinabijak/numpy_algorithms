import itertools
from helper_functions import *


class OvAKernelPerceptron:
    """
    One-vs-All Kernel Perceptron.
    :param kernel: kernel type, "p" for polynomial, "g" for gaussian.
    :param param: parameter of the kernel; degree of polynomial, or beta value of gaussian.
    """

    def __init__(self, kernel, param):
        self.param = param
        self.kernel = kernel
        self.w = None
        self.alphas = None
        self.x = None
        self.y = None
        self.test_mistakes = None

    def train(self, train_set):

        # number of examples
        m = train_set.shape[0]
        train_y = train_set[:, 0]
        train_x = train_set[:, 1:]

        # Alphas store values of alpha for each class
        alphas = np.zeros(shape=(m, 10))
        train_gram = grammatrix(train_x, train_x, self.kernel, self.param)
        # initialize w to store indexes of train examples that have non-zero alpha
        w = []
        # keep track of the best valid set error rate to stop training.
        best_train_err = 1.0
        best_w = []
        cont = True
        while cont:
            train_incorrect = 0
            for index in range(m):
                preds = np.einsum("ij, i", alphas[w, :], train_gram[index, w])

                y = [1 if j == train_y[index] else -1 for j in range(10)] * preds <= 0
                if np.any(y):
                    y = y.astype(int)
                    alphas[index, :] -= y * mysign(preds)
                    w.append(index) if index not in w else w

                y_hat = np.argmax(preds)

                if y_hat != train_y[index]:
                    train_incorrect += 1
            train_err = train_incorrect / len(train_x)
            if train_err == 0:
                cont = False
                best_w = w
                best_train_err = train_err
            elif train_err <= best_train_err:
                cont = True
                best_w = w
                best_train_err = train_err
            else:
                cont = False

        self.alphas = alphas[best_w, :]
        self.x = train_x[best_w, :]
        self.y = train_y[best_w]
        self.w = best_w

        return best_train_err

    def predict(self, data, confusion_matrix=False):
        """
        :param confusion_matrix:
        :param data:
        :return: train error
        """
        test_y = data[:, 0]
        gram = grammatrix(data[:, 1:], self.x, self.kernel, self.param)
        incorrect = 0
        mistaken = []
        if confusion_matrix:
            cm = np.zeros(shape=(10, 10))

        for index in range(len(test_y)):
            predictions = np.einsum("ij, i", self.alphas, gram[index, :])
            y_pred = int(np.argmax(predictions))
            if y_pred != test_y[index]:
                incorrect += 1
                mistaken.append(index)
                if confusion_matrix:
                    cm[int(test_y[index]), y_pred] += 1

        err = incorrect / len(test_y)
        self.test_mistakes = mistaken
        if confusion_matrix:
            return err, cm.astype(int)
        return err


class OvOKernelPerceptron:

    def __init__(self, kernel, param):
        self.param = param
        self.kernel = kernel
        self.predictor_indices = None
        self.alphas = None
        self.x = None
        self.y = None
        self.test_mistakes = None

    def train(self, trainset):

        m = trainset.shape[0]
        train_y = trainset[:, 0]
        train_x = trainset[:, 1:]

        combs = list(itertools.combinations(np.arange(10).astype(int), 2))
        alphas = {i: np.zeros(shape=(train_x.shape[0])) for i in combs}
        gram = grammatrix(train_x, train_x, self.kernel, self.param)
        w = []
        prev_w = []
        predictor_indices = get_indices(train_y, alphas)
        best_train_err = 1.0
        cont = True
        while cont:
            incorrect = 0
            ties = 0
            for index in range(m):
                # get predictions:

                model_outputs = np.zeros(45)
                preds = np.zeros(10)
                for i, alpha in enumerate(alphas.items()):
                    model_outputs[i] = np.dot(alpha[1][predictor_indices[i]], gram[index, predictor_indices[i]])
                    binary_pred = my_sign(model_outputs[i])
                    preds[alpha[0][classify(binary_pred)]] += 1

                # mask:
                y = [-1 if alpha[0] == train_y[index] else 1 if alpha[1] == train_y[index] else 0 for alpha in alphas]
                check_y = y * model_outputs <= 0
                # update weights
                if np.any(check_y):
                    w.append(index) if index not in w else w
                    for i, alpha in enumerate(alphas.values()):
                        alpha[index] -= check_y[i] * my_sign(model_outputs[i])

                winners = np.argwhere(preds == np.amax(preds)).flatten()
                if len(winners) != 1:
                    ties += 1
                    coms = list(itertools.combinations(winners, 2))
                    predicted = np.zeros(10)
                    for i, alpha in enumerate(alphas):
                        for com in coms:
                            if alpha == com:
                                bin_pred = my_sign(model_outputs[i])
                                predicted[com[classify(bin_pred)]] += 1
                    y_pred = np.argmax(predicted)
                else:
                    y_pred = np.argmax(preds)

                if y_pred != train_y[index]:
                    incorrect += 1

            train_err = incorrect / len(train_x)
            if train_err < best_train_err:
                cont = True
                prev_w = w
                best_train_err = train_err
            else:
                cont = False

        self.x = train_x[prev_w]
        self.y = train_y[prev_w]
        self.alphas = alphas
        self.predictor_indices = get_indices(self.y, self.alphas)

        return best_train_err

    def predict(self, data):
        test_y = data[:, 0]
        test_gram = grammatrix(data[:, 1:], self.x, self.kernel, self.param)
        incorrect = 0
        mistaken = []
        for index in range(len(test_y)):
            model_outputs = np.zeros(45)
            preds = np.zeros(10)
            for i, alpha in enumerate(self.alphas.items()):
                model_outputs[i] = np.dot(alpha[1][self.predictor_indices[i]],
                                          test_gram[index, self.predictor_indices[i]])
                binary_pred = my_sign(model_outputs[i])
                preds[alpha[0][classify(binary_pred)]] += 1

            y_pred = np.argmax(preds)

            if y_pred != test_y[index]:
                incorrect += 1
                mistaken.append(index)

        test_err = incorrect / len(test_y)
        self.test_mistakes = mistaken

        return test_err
