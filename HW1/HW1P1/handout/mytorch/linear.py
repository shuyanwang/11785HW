# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
from typing import Optional

import numpy as np
import math


class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):
        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.momentum_W = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

        self.X: Optional[np.ndarray] = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.X = x
        return x @ self.W + self.b

    def backward(self, delta: np.ndarray):
        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # raise NotImplemented
        # y_bo = \sum_i^IN (x_bi*w_io) + b_o
        # dy_bo/dW_io = x_bi
        # dL/dy_bo = delta_bo
        # dL/dW_io = x_bi*delta_bo via b in batch
        # average on batch, because in parallel.
        B = self.X.shape[0]
        I = self.X.shape[1]
        O = delta.shape[1]
        self.dW = np.mean(self.X.reshape((B, I, 1)) * delta.reshape((B, 1, O)), axis=0)

        # dy_bo/db_o = 1
        # dL/db_o = delta_bo via b in batch -> mean

        self.db = np.mean(delta, axis=0).reshape(1, -1)

        # dy_bo/dx_bi = w_io
        # where is x_bi used? in y_bO for all O, so we should sum over O to account for all pDs

        dX = delta.reshape((B, 1, O)) * self.W.reshape((1, I, O))
        dX = np.sum(dX, axis=2)

        return dX


if __name__ == '__main__':
    def init_func(dim1, dim2=None):
        if dim2 is None:
            return np.random.random((1, dim1))
        else:
            return np.random.random((dim1, dim2))


    x = np.random.random((10, 5))

    l = Linear(5, 3, init_func, init_func)

    y = l(x)

    t = 1
