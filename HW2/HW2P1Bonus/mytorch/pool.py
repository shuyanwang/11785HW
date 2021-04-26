import numpy as np


class MaxPoolLayer:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.used = None
        self.in_w = None
        self.in_h = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.in_w, self.in_h = x.shape[2], x.shape[3]
        out_w, out_h = (self.in_w - self.kernel) // self.stride + 1, (
                self.in_h - self.kernel) // self.stride + 1

        out = np.zeros((x.shape[0], x.shape[1], out_w, out_h))
        self.used = np.zeros_like(out, dtype=int)

        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for col in range(out_w):
                    for row in range(out_h):
                        max_id = np.argmax(
                                x[b, c, col * self.stride:col * self.stride + self.kernel,
                                row * self.stride:row * self.stride + self.kernel])
                        self.used[b, c, col, row] = max_id
                        id = np.unravel_index(max_id, (self.kernel, self.kernel))
                        out[b, c, col, row] = x[
                            b, c, col * self.stride + id[0], row * self.stride + id[1]]

        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        B, C = delta.shape[0], delta.shape[1]
        dx = np.zeros((B, C, self.in_w, self.in_h))
        for b in range(B):
            for c in range(C):
                for col in range(delta.shape[2]):
                    for row in range(delta.shape[3]):
                        id = np.unravel_index(self.used[b, c, col, row],
                                              (self.kernel, self.kernel))
                        dx[b, c, col * self.stride + id[0], row * self.stride + id[1]] = delta[
                            b, c, col, row]

        return dx


class MeanPoolLayer:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.in_w = None
        self.in_h = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.in_w, self.in_h = x.shape[2], x.shape[3]
        out_w, out_h = (self.in_w - self.kernel) // self.stride + 1, (
                self.in_h - self.kernel) // self.stride + 1

        out = np.zeros((x.shape[0], x.shape[1], out_w, out_h))

        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for col in range(out_w):
                    for row in range(out_h):
                        out[b, c, col, row] = np.mean(
                                x[b, c, col * self.stride:col * self.stride + self.kernel,
                                row * self.stride:row * self.stride + self.kernel])

        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        B, C = delta.shape[0], delta.shape[1]
        dx = np.zeros((B, C, self.in_w, self.in_h))
        for b in range(B):
            for c in range(C):
                for col in range(delta.shape[2]):
                    for row in range(delta.shape[3]):
                        for k1 in range(self.kernel):
                            for k2 in range(self.kernel):
                                dx[b, c, col * self.stride + k1, row * self.stride + k2] += delta[
                                                                                                b, c, col, row] / self.kernel / self.kernel

        return dx
