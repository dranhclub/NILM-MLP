import numpy as np
from math import exp
from scipy.special import xlogy
from scipy.special import expit as logistic_sigmoid

np.random.seed(0)


def sigmoid(x):
    return logistic_sigmoid(x)


def sigmoid_deriv(x):
    sm = sigmoid(x)
    return sm * (1 - sm)


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return 1 if x > 0 else 0


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.exp2(tanh(x))


def binary_log_loss(y_true, y_prob):
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -(xlogy(y_true, y_prob).sum() +
             xlogy(1 - y_true, 1 - y_prob).sum()) / y_prob.shape[0]


class MLPClassifier():
    def __init__(self):
        pass

    def fit(self, X, y):
        n_sample, n_feature = X.shape

        n_input_unit = n_feature
        n_hidden_unit = 30
        n_output_unit = y.shape[1]

        W1 = np.random.uniform(-1, 1, (n_hidden_unit, n_input_unit))
        W2 = np.random.uniform(-1, 1, (n_output_unit, n_hidden_unit))
        b1 = np.random.uniform(-1, 1, (n_hidden_unit, 1))
        b2 = np.random.uniform(-1, 1, (n_output_unit, 1))

        learning_rate = 0.01
        hidden_activation_func = tanh
        hidden_activation_deriv = tanh_deriv
        output_activation_func = sigmoid

        max_iter = 100

        for it in range(max_iter):
            print("iteration=", it)
            A = np.zeros((3, n_sample))  # activations
            A = [X] + [None] + [None]

            # forward pass
            Z1 = np.dot(W1, X.T) + b1
            A[1] = hidden_activation_func(Z1)
            Z2 = np.dot(W2, A[1]) + b2
            A[2] = output_activation_func(Z2)

            # compute loss (cross entropy or log loss)
            log_loss = binary_log_loss(y, A[2].T)
            print(f'{log_loss=}')

            # backpropagation
            dZ2 = A[2] - y.T
            dW2 = np.dot(dZ2, A[1].T) / n_sample
            db2 = np.sum(dZ2, axis=1, keepdims=True) / n_sample
            dZ1 = np.multiply(np.dot(W2.T, dZ2), hidden_activation_deriv(Z1))
            dW1 = np.dot(dZ1, X)
            db1 = np.sum(dZ1, axis=1, keepdims=True) / n_sample

            # update params
            W1 = W1 - learning_rate * dW1
            W2 = W2 - learning_rate * dW2
            b1 = b1 - learning_rate * db1
            b2 = b2 - learning_rate * db2

        print("End training")
        self.n_input_unit_ = n_input_unit
        self.n_hidden_unit_ = n_hidden_unit
        self.n_output_unit_ = n_output_unit
        self.W1_ = W1
        self.W2_ = W2
        self.b1_ = b1
        self.b2_ = b2

    def predict(self, X):
        W1 = self.W1_
        W2 = self.W2_
        b1 = self.b1_
        b2 = self.b2_

        Z1 = np.dot(W1, X.T) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        ret = A2.T
        ret[ret >= 0.5] = 1
        ret[ret < 0.5] = 0
        return ret.astype(int)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        result = (y_pred == y_test).all(axis=1)
        return result[result == True].shape[0] / y_test.shape[0]