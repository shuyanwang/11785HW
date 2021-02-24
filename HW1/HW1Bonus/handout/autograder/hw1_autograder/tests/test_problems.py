import numpy as np
import os
from helpers.helpers import *
import sys
import pickle

base_dir = 'autograder/hw1_autograder/'

autolab = bool(int(os.environ['AUTOLAB'])) if 'AUTOLAB' in os.environ.keys() else False
saved_data = pickle.load(open(base_dir + "data.pkl", 'rb'))
rtol = 1e-4
atol = 1e-04
TOLERANCE = 1e-4

SEED = 2019
if autolab:
    print("We are on Autolab")
    TRAINDATAPATH = "/datasets/11785/mnist_train.csv"
    TESTDATAPATH = "/datasets/11785/mnist_test.csv"
    sys.path.append('handin/')
else:
    print("We are on local")
    TRAINDATAPATH = base_dir + "tests/data/mnist_train.csv"
    TESTDATAPATH = base_dir + "tests/data/mnist_test.csv"

if os.path.exists(TRAINDATAPATH):
    print("Train data exists")
if os.path.exists(TESTDATAPATH):
    print("Test data exists")

sys.path.append('mytorch')
import activation
import loss
import linear
import batchnorm
import optimizer
import dropout

sys.path.append('hw1')
import hw1
# import mc


def raw_mnist(path):
    return (cleaned_mnist(path))


def cleaned_mnist(path):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:, 1:]
    Y = data[:, 0]
    Y = Y.astype(int)
    return X, Y


def reset_prng():
    np.random.seed(11785)


def weight_init(x, y):
    return np.random.randn(x, y)


def bias_init(x):
    return np.zeros((1, x))


## Test For ADAM:
def test_adam():
    data = saved_data[21]
    assert len(data) == 8
    x = data[0]
    y = data[1]
    #saved_data = pickle.load(open(base_dir + "data.pkl", 'rb'))
    solW = pickle.load(open(base_dir + "adam_solW.pkl", "rb"))
    solb = pickle.load(open(base_dir + "adam_solb.pkl", "rb"))

    reset_prng()
    mlp = hw1.MLP(784, 10, [64, 32], [activation.Sigmoid(), activation.Sigmoid(), activation.Identity()], weight_init, bias_init, loss.SoftmaxCrossEntropy(), 0.008,
                  momentum=0.856, num_bn_layers=0)

    num_test_updates = 5
    optim = optimizer.adam(mlp, 0.9, 0.999)
    for u in range(num_test_updates):
        mlp.zero_grads()
        mlp.forward(x)
        mlp.backward(y)
        optim.step()
    mlp.eval()

    W = [x.W for x in mlp.linear_layers]
    b = [x.b for x in mlp.linear_layers]

    for i, (pred, gt) in enumerate(zip(W, solW)):
        closeness_test(pred, gt, "mlp.linear_layers[%d].W" % i)

    for i, (pred, gt) in enumerate(zip(b, solb)):
        closeness_test(pred, gt, "mlp.linear_layers[%d].b" % i)


#Test for Dropout forward:
def test_dropout_forward():
    reset_prng()
    x = np.random.randn(20, 64)
    reset_prng()
    dropout_layer = dropout.Dropout(p = 0.5)
    y = dropout_layer(x)
    soly = pickle.load(open(base_dir + "sol_dropout_forward.pkl", "rb"))
    
    """
    output = open(base_dir+'sol_dropout_forward.pkl', 'wb')
    pickle.dump(y, output)
    output.close()
    
    output = open(base_dir+'solb.pkl', 'wb')
    pickle.dump(b, output)
    output.close()
    """

    closeness_test(y, soly, "dropout.forward(x)")


def test_dropout_backward():
    reset_prng()
    x = np.random.randn(20, 64)
    reset_prng()
    dropout_layer = dropout.Dropout(p = 0.5)
    y = dropout_layer(x)
    reset_prng()
    delta = np.random.randn(20, 64)
    dx = dropout_layer.backward(delta)
    dx_sol = pickle.load(open(base_dir + "sol_dropout_backward.pkl", "rb"))
    closeness_test(dx, dx_sol, "dropout.backward(x)")

def failed_test_names(names, preds, gts, status):
    values = [(preds[i], gts[i]) for i, s in enumerate(status) if not s]
    names = [n for n, s in zip(names, status) if not s]
    return names, values


def union(xs, ys):
    return [x or y for x, y in zip(xs, ys)]


def assert_any_zeros(nparr):
    for i in range(len(nparr)):
        assert (np.all(nparr[i], 0))
