from helper_functions import *


class Three_Layer_MLP:

    def __init__(self, hid_dims):
        self.hid_dims = hid_dims
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None

    def train(self, trainset, n_epochs, lr):

        train_x = trainset[:, 1:]
        # dimensionality of the data
        n = train_x.shape[1]
        # initialize weights and biases
        W1 = np.random.uniform(low=-0.3, high=0.3, size=(n, self.hid_dims[0]))
        W2 = np.random.uniform(low=-0.3, high=0.3, size=(self.hid_dims[0], self.hid_dims[1]))
        W3 = np.random.uniform(low=-0.3, high=0.3, size=(self.hid_dims[1], 10))

        b1 = np.zeros(shape=(self.hid_dims[0], 1))
        b2 = np.zeros(shape=(self.hid_dims[1], 1))
        b3 = np.zeros(shape=(10, 1))

        # train loop
        hist = {"loss": [], "acc": []}
        for epoch in range(n_epochs):
            epoch_loss = 0
            correct = 0
            np.random.shuffle(trainset)
            train_y = trainset[:, 0]
            train_x = trainset[:, 1:]

            for X, y in zip(train_x, train_y):
                # FORWARD
                z1 = np.dot(X.T, W1).reshape(-1, 1) + b1
                a1 = ReLU(z1)
                z2 = np.dot(a1.T, W2).reshape(-1, 1) + b2
                a2 = ReLU(z2)
                z3 = np.dot(a2.T, W3).reshape(-1, 1) + b3
                a3 = softmax(z3)

                t_one_hot = np.zeros(shape=(10, 1))
                t_one_hot[int(y)] = 1

                # BACKWARD
                del3 = a3 - t_one_hot
                del2 = np.dot(W3, del3) * ReLU_prime(z2)
                del1 = np.dot(W2, del2) * ReLU_prime(z1)

                grad_W3 = np.dot(a2, del3.T)
                grad_b3 = del3
                grad_W2 = np.dot(a1, del2.T)
                grad_b2 = del2
                grad_W1 = np.dot(X.reshape(-1, 1), del1.T)
                grad_b1 = del1

                y_pred = np.argmax(a3)
                if y_pred == y:
                    correct += 1
                epoch_loss += loss(y, a3)

                # UPDATE WEIGHTS
                W1 -= lr * grad_W1
                b1 -= lr * grad_b1
                W2 -= lr * grad_W2
                b2 -= lr * grad_b2
                W3 -= lr * grad_W3
                b3 -= lr * grad_b3

            acc = correct / len(train_x)
            epoch_loss = epoch_loss / len(train_x)
            hist["acc"].append(acc)
            hist["loss"].append(epoch_loss)
            print(f"epoch: {epoch + 1}, loss: {epoch_loss}, accuracy: {acc}")

        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        return hist

    def predict(self, testset):
        test_y = testset[:, 0]
        test_x = testset[:, 1:]

        incorrect = 0
        for X, y in zip(test_x, test_y):

            z1 = np.dot(X.T, self.W1).reshape(-1, 1) + self.b1
            a1 = ReLU(z1)
            z2 = np.dot(a1.T, self.W2).reshape(-1, 1) + self.b2
            a2 = ReLU(z2)
            z3 = np.dot(a2.T, self.W3).reshape(-1, 1) + self.b3
            a3 = softmax(z3)

            y_pred = np.argmax(a3)

            if y_pred != y:
                incorrect += 1
        err = incorrect / len(test_x)
        print(err)
        return err
