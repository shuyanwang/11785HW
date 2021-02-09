"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *
from typing import List


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations: List[Activation],
                 weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = []
        if len(hiddens) == 0:
            self.linear_layers.append(Linear(input_size, output_size, weight_init_fn, bias_init_fn))
        else:
            self.linear_layers.append(Linear(input_size, hiddens[0], weight_init_fn, bias_init_fn))

        for i, o in zip(hiddens[0:-1], hiddens[1:]):
            self.linear_layers.append(Linear(i, o, weight_init_fn, bias_init_fn))
        self.linear_layers.append(Linear(hiddens[-1], output_size, weight_init_fn, bias_init_fn))

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            # if self.num_bn_layers < len(hiddens):
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))
            # else:
            #     for i in range(len(hiddens)):
            #         self.bn_layers.append(BatchNorm(hiddens[i]))
            #     self.bn_layers.append(BatchNorm(output_size))

        self.output = None

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        # raise NotImplemented

        layer_id = 0
        while layer_id < self.num_bn_layers:
            x = self.activations[layer_id](
                self.bn_layers[layer_id](self.linear_layers[layer_id](x), eval=not self.train_mode))
        while layer_id < self.nlayers:
            x = self.activations[layer_id](self.linear_layers[layer_id](x))
        self.output = x
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for linear in self.linear_layers:
            linear.db.fill(0.0)
            linear.dW.fill(0.0)

        for bn in self.bn_layers:
            bn.dbeta.fill(0.0)
            bn.dgamma.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for linear in self.linear_layers:
            linear.momentum_W = self.momentum * linear.momentum_W - self.lr * linear.dW
            linear.W += linear.momentum_W

            linear.momentum_b = self.momentum * linear.momentum_b - self.lr * linear.db
            linear.b += linear.momentum_b

        for bn in self.bn_layers:
            bn.gamma -= bn.dgamma * self.lr
            bn.beta -= bn.dbeta * self.lr

        # raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        # raise NotImplemented
        # I do not think labels should be used here.

        gradient = self.criterion.derivative()

        for layer in range(self.nlayers - 1, self.nlayers - self.num_bn_layers - 1, -1):
            gradient = gradient * self.activations[layer].derivative()
            gradient = self.linear_layers[layer].backward(gradient)

        for layer in range(self.nlayers - self.num_bn_layers - 1, -1, -1):
            gradient = gradient * self.activations[layer].derivative()
            gradient = self.bn_layers[layer].backward(gradient)
            gradient = self.linear_layers[layer].backward(gradient)

    def error(self, labels):
        return (np.argmax(self.output, axis=1) != np.argmax(labels, axis=1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


# This function does not carry any points. You can try and complete this function to train your
# network.
def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):
            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):
            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
