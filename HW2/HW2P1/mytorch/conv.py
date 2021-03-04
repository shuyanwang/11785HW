# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
from typing import Optional

import numpy as np


class Conv1D:
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x: Optional[np.ndarray] = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        # W: (out_channel,in_channel,kernel_size)
        self.x = x
        batch_size, input_size = x.shape[0], x.shape[2]
        kernel_size = self.W.shape[2]
        output_size = (input_size - kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, output_size))

        for o in range(output_size):
            x_filter = x[:, :, o * self.stride:o * self.stride + kernel_size]  # (batch,cin,k)
            out[:, :, o] = np.tensordot(x_filter, self.W, ([1, 2], [1, 2]))  # (batch,out_channel)
            # out[:, :, o] = np.einsum('bik,oik->bo', x_filter, self.W) # also works

        out += np.reshape(self.b, (self.b.shape[0], 1))

        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, input_size = self.x.shape[0], self.x.shape[2]
        kernel_size = self.W.shape[2]
        output_size = delta.shape[2]

        # self.dW[:] = 0
        # self.db[:] = 0

        w_flipped = np.flip(self.W, axis=2)  # (cout,cin,k)
        dx = np.zeros_like(self.x)

        pad_size_for_x = input_size + (kernel_size - 1)
        # (output_size - 1) * self.stride + 1 + 2 * (kernel_size - 1) WRONG
        pad_size_for_w = (output_size - 1) * self.stride + 1

        delta_padded_for_x = np.zeros((batch_size, self.out_channel, pad_size_for_x))
        delta_padded_for_w = np.zeros((batch_size, self.out_channel, pad_size_for_w))

        for o in range(output_size):
            delta_padded_for_x[:, :, o * self.stride + (kernel_size - 1)] = delta[:, :, o]
            delta_padded_for_w[:, :, o * self.stride] = delta[:, :, o]

        for i in range(input_size):
            delta_filter_for_x = delta_padded_for_x[:, :, i:i + kernel_size]  # (batch,cout,k)
            # w_flipped: (cout,cin,kernel_size)

            assert delta_filter_for_x.shape[2] == w_flipped.shape[2]

            dx[:, :, i] = np.tensordot(delta_filter_for_x, w_flipped, ([1, 2], [0, 2]))
            # (batch,cin)

        for k in range(kernel_size):
            # delta_pad_w: (batch,cout,pad_w)
            x_filter_for_w = self.x[:, :, k:k + pad_size_for_w]  # (batch,cin,pad_w)
            self.dW[:, :, k] = np.tensordot(delta_padded_for_w, x_filter_for_w, ([0, 2], [0, 2]))
            # (out_channel,in_channel,kernel_size)

        d = np.sum(delta, axis=2)
        d = np.sum(d, axis=0)
        self.db = d

        return dx


class Conv2D:
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.x: Optional[np.ndarray] = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # w: (cout,cin,k,k)

        self.x = x
        batch_size, in_w, in_h = x.shape[0], x.shape[2], x.shape[3]
        kernel_size = self.W.shape[2]
        out_w = (in_w - kernel_size) // self.stride + 1
        out_h = (in_h - kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, out_w, out_h))

        for col in range(out_w):
            for row in range(out_h):
                x_filter = x[:, :, col * self.stride:col * self.stride + kernel_size,
                           row * self.stride:row * self.stride + kernel_size]  # (batch,cin,k)
                out[:, :, col, row] = np.tensordot(x_filter, self.W, ([1, 2, 3], [1, 2, 3]))
                # (batch,out_channel)

        out += np.reshape(self.b, (self.b.shape[0], 1, 1))

        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, in_w, in_h = self.x.shape[0], self.x.shape[2], self.x.shape[3]
        kernel_size = self.W.shape[2]
        out_w, out_h = delta.shape[2], delta.shape[3]

        w_flipped = np.flip(self.W, axis=(2, 3))  # (cout,cin,k,k)
        dx = np.zeros_like(self.x)

        pad_width_for_x = in_w + (kernel_size - 1)
        pad_height_for_x = in_h + (kernel_size - 1)
        # (output_size - 1) * self.stride + 1 + 2 * (kernel_size - 1) WRONG
        pad_width_for_w = (out_w - 1) * self.stride + 1
        pad_height_for_w = (out_h - 1) * self.stride + 1

        delta_padded_for_x = np.zeros(
                (batch_size, self.out_channel, pad_width_for_x, pad_height_for_x))
        delta_padded_for_w = np.zeros(
                (batch_size, self.out_channel, pad_width_for_w, pad_height_for_w))

        for col in range(out_w):
            for row in range(out_h):
                delta_padded_for_x[:, :, col * self.stride + (kernel_size - 1),
                row * self.stride + (kernel_size - 1)] = delta[:, :, col, row]
                delta_padded_for_w[:, :, col * self.stride,
                row * self.stride] = delta[:, :, col, row]

        for col in range(in_w):
            for row in range(in_h):
                delta_filter_for_x = delta_padded_for_x[:, :, col:col + kernel_size,
                                     row:row + kernel_size]  # (batch,cout,k)
                # w_flipped: (cout,cin,kernel_size)

                assert delta_filter_for_x.shape[2] == w_flipped.shape[2]

                dx[:, :, col, row] = np.tensordot(delta_filter_for_x, w_flipped,
                                                  ([1, 2, 3], [0, 2, 3]))
                # (batch,cin)

        for k1 in range(kernel_size):
            for k2 in range(kernel_size):
                # delta_pad_w: (batch,cout,pad_w)
                x_filter_for_w = self.x[:, :, k1:k1 + pad_width_for_w, k2:k2 + pad_height_for_w]
                self.dW[:, :, k1, k2] = np.tensordot(delta_padded_for_w, x_filter_for_w,
                                                     ([0, 2, 3], [0, 2, 3]))
                # (out_channel,in_channel,kernel_size)
        d = np.sum(delta, axis=3)
        d = np.sum(d, axis=2)
        d = np.sum(d, axis=0)
        self.db = d

        return dx


class Flatten:
    def __init__(self):
        self.b = None
        self.c = None
        self.w = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return np.reshape(x, (self.b, -1))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return np.reshape(delta, (self.b, self.c, self.w))
