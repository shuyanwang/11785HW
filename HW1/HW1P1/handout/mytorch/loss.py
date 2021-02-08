# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os


def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """
    Log sum.
    :param x: (batch, M)
    :return: (batch,)
    """
    a = np.max(x, axis=1)
    return a + np.log(np.sum(np.exp(x - a), axis=1))


# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.prediction = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        # log(sigma) = log(e^x)-log_sum = x-log_sum
        # so sigma = exp(x-log_sum)
        log_sigma = x - log_sum_exp(x)  # batch*10
        self.prediction = np.exp(log_sigma)  # batch*10
        self.loss = -np.sum(y * log_sigma, axis=1)
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        return self.prediction - self.labels
