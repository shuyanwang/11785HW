import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        self.r = None
        self.z = None
        self.n = None
        self.x = None
        self.hidden = None

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        x = np.expand_dims(x, 1)  # (d, 1)
        h = np.expand_dims(h, 1)  # (h, 1)

        # a = np.matmul(self.Wrx, x) - self.Wrx @ x

        self.r = self.r_act(
                self.Wrx @ x + self.bir.reshape((-1, 1)) + self.Wrh @ h + self.bhr.reshape(-1, 1))
        self.z = self.z_act(
                self.Wzx @ x + self.biz.reshape((-1, 1)) + self.Wzh @ h + self.bhz.reshape((-1, 1)))
        self.n = self.h_act(self.Wnx @ x + self.bin.reshape((-1, 1)) + self.r * (
                self.Wnh @ h + self.bhn.reshape(-1, 1)))
        h_t = (1 - self.z) * self.n + self.z * h

        self.r = np.squeeze(self.r)
        self.z = np.squeeze(self.z)
        self.n = np.squeeze(self.n)
        h_t = np.squeeze(h_t)

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        delta = np.reshape(delta, (-1, 1))
        # delta (h,1)

        r = np.reshape(self.r, (-1, 1))
        z = np.reshape(self.z, (-1, 1))
        n = np.reshape(self.n, (-1, 1))
        h_prev = np.reshape(self.hidden, (-1, 1))
        x = np.reshape(self.x, (-1, 1)).transpose()  # (1,d)

        dn = delta * (1 - z)
        dz = delta * (-n + h_prev)

        d_n_affine = dn * self.h_act.derivative(n)  # (n,1)
        self.dWnx = d_n_affine @ x
        self.dbin = np.squeeze(d_n_affine)
        dr = d_n_affine * (self.Wnh @ (np.expand_dims(self.hidden, 1)) + self.bhn.reshape(-1, 1))
        self.dWnh = d_n_affine * r @ h_prev.transpose()
        self.dbhn = np.squeeze(d_n_affine * r)

        dz_affine = dz * self.z_act.derivative()
        self.dWzx = dz_affine @ x
        self.dbiz = np.squeeze(dz_affine)
        self.dWzh = dz_affine @ h_prev.transpose()
        self.dbhz = np.squeeze(dz_affine)

        dr_affine = dr * self.r_act.derivative()
        self.dWrx = dr_affine @ x
        self.dbir = np.squeeze(dr_affine)
        self.dWrh = dr_affine @ h_prev.transpose()
        self.dbhr = np.squeeze(dr_affine)

        dx = np.zeros((1, self.d))
        dx += d_n_affine.transpose() @ self.Wnx
        dx += dz_affine.transpose() @ self.Wzx
        dx += dr_affine.transpose() @ self.Wrx

        dh_prev = np.zeros((1, self.h))
        dh_prev += (delta * z).transpose()
        dh_prev += (d_n_affine * r).transpose() @ self.Wnh
        dh_prev += dz_affine.transpose() @ self.Wzh
        dh_prev += dr_affine.transpose() @ self.Wrh

        return dx, dh_prev
