import numpy as np
from copy import deepcopy

from kNN import KNNClassifier


def train_test_split(data, train_size=0.75, k=False):
    """
    :param data: sequence of the data where length: shape[0] defines the number of samples.
    :param train_size: should be between 0.0 and 1.0. Represents the proportion of data to include in the train split.
                        Test split size is chosen to complement the train_size.
    :param k: if True, further splits train data into 5-folds.
    :type k: bool
    :return:
    """
    if train_size > 1:
        raise ValueError(
            "Choose train_size between 0.0 and 1.0 to represent the proportion of data to include in the train split")

    new_data = np.copy(data)
    np.random.shuffle(new_data)
    n_arrays = len(new_data)

    border = int(np.ceil(train_size * n_arrays))

    train = new_data[:border, :]
    test = new_data[border:, :]

    if k:
        train_size = len(train)
        base_fold_size = np.floor(train_size / 5)
        reminder = int(train_size % 5)
        sizes = np.ones(shape=5) * base_fold_size
        sizes[:reminder] += 1
        indices = np.cumsum(sizes).astype(int)

        k_train = [0 for i in range(5)]
        for idx in range(5):
            if idx == 0:
                k_train[0] = train[:indices[0], :]
            elif idx == 4:
                k_train[4] = train[indices[3]:, :]
            else:
                k_train[idx] = train[indices[idx - 1]:indices[idx], :]

        return k_train, train, test

    return train, test


def CV(model, data, kernel, param_values):
    """
    cross valdation of the model on the data
    :param model: OvOKernelPerceptron or OvAKernelPerceptron
    :return: CV error and the data split used.
    """
    k_fold, main_train, main_test = train_test_split(data, 0.8, True)

    # k is the index of the train split we leave as valid set
    errors = np.zeros(shape=(len(param_values), 5))
    for k in range(5):
        fold = deepcopy(k_fold)
        valid_set = fold[k]
        fold.pop(k)
        train_set = np.concatenate(fold, axis=0)
        for index, d in enumerate(param_values):
            if type(model) == KNNClassifier:
                val_err = model.predict(train_set, d, valid_set)
            else:
                cur_model = model(kernel, d)
                cur_model.train(train_set)
                val_err = cur_model.predict(valid_set)
            errors[index, k] = val_err

    return errors, (main_train, main_test)
