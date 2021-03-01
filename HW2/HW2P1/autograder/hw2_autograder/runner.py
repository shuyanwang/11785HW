import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import multiprocessing as mtp
import traceback
import sys
from helpers import *

sys.path.append('mytorch')

from test import *
from loss import *
from activation import *
from batchnorm import *
from linear import *
from conv import *

sys.path.append('hw2')
import mc

np.random.seed(2020)
############################################################################################
################################   Section 2 - MCQ    ######################################
############################################################################################

def test_mcq_1():
    return 'b' == mc.question_1()
def test_mcq_2():
    return 'd' == mc.question_2()
def test_mcq_3():
    return 'b' == mc.question_3()
def test_mcq_4():
    return 'a' == mc.question_4()
def test_mcq_5():
    res = 'a' == mc.question_5()        
    print('-'*20)
    return res 
############################################################################################
####################################   Section 3.1    ######################################
############################################################################################

def test_cnn_correctness_once(idx):

    scores_dict = [0,0,0,0]

    ############################################################################################
    #############################   Initialize hyperparameters    ##############################
    ############################################################################################
    rint = np.random.randint
    norm = np.linalg.norm
    in_c, out_c = rint(5,15), rint(5,15)
    kernel, stride =  rint(1,10), rint(1,10)
    batch, width = rint(1,4), rint(20,300)



    def info():
        print('\nTesting model:')
        print('    in_channel: {}, out_channel: {},'.format(in_c,out_c))
        print('    kernel size: {}, stride: {},'.format(kernel,stride))
        print('    batch size: {}, input size: {}.'.format(batch,width))


    ##############################################################################################
    ##########    Initialize the CNN layer and copy parameters to a PyTorch CNN layer   ##########
    ##############################################################################################
    def random_normal_weight_init_fn(out_channel, in_channel, kernel_size):
        return np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
    
    if idx == 0:
        print("---Conv1D Forward (Tests)---")
    try:
        net = Conv1D(in_c, out_c, kernel, stride, random_normal_weight_init_fn, np.zeros)
    except:
        info()
        print('Failed to pass parameters to your Conv1D function!')
        return scores_dict

    model = nn.Conv1d(in_c, out_c, kernel, stride)
    model.weight = nn.Parameter(torch.tensor(net.W))
    model.bias = nn.Parameter(torch.tensor(net.b))


    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x = np.random.randn(batch, in_c, width)
    x1 = Variable(torch.tensor(x),requires_grad=True)
    y1 = model(x1)
    b, c, w = y1.shape
    delta = np.random.randn(b,c,w)
    y1.backward(torch.tensor(delta))


    #############################################################################################
    ##########################    Get your forward results and compare ##########################
    #############################################################################################
    y = net(x)
    assert y.shape == y1.shape
    if not(y.shape == y1.shape): print("FAILURE")


    forward_res = y - y1.detach().numpy()
    forward_res_norm = abs(forward_res).max()



    if forward_res_norm < 1e-12:
        scores_dict[0] =  1

    else:
        info()
        print('Fail to return correct forward values')
        return scores_dict

    #############################################################################################
    ##################   Get your backward results and check the tensor shape ###################
    #############################################################################################
    dx = net.backward(delta)

    if idx == 0:
        print("---Conv1D Backward (Tests)---")
    assert dx.shape == x.shape    
    assert net.dW.shape == model.weight.grad.detach().numpy().shape
    assert net.db.shape == model.bias.grad.detach().numpy().shape
    #############################################################################################
    ################   Check your dx, dW and db with PyTorch build-in functions #################
    #############################################################################################
    dx1 = x1.grad.detach().numpy()
    delta_res_norm = abs(dx - dx1).max()

    dW_res = net.dW - model.weight.grad.detach().numpy()
    dW_res_norm = abs(dW_res).max()

    db_res = net.db - model.bias.grad.detach().numpy()
    db_res_norm = abs(db_res).max()


    if delta_res_norm < 1e-12:
        scores_dict[1] = 1
    
    if dW_res_norm < 1e-12:
        scores_dict[2] = 1

    if db_res_norm < 1e-12:
        scores_dict[3] = 1

    if min(scores_dict) != 1:
        info()
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dx')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values db')
    return scores_dict


def test_cnn_correctness():
    scores = []
    worker = min(mtp.cpu_count(),4)
    p = mtp.Pool(worker)
    
    for __ in range(15):
        scores_dict = test_cnn_correctness_once(__) 
        scores.append(scores_dict)
        if min(scores_dict) != 1:
            return False
    
    # scores = np.array(scores).min(0)
    a, b, c, d = np.array(scores).min(0)
    print('Section 4.1.1 - Forward | 15 points')
    print('Conv1D Forward:', 'PASS' if a == 1 else 'FAIL')

    print('Section 4.1.2 - Backward | 25 points')
    print('Conv1D dX:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1D dW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1D db:', 'PASS' if d == 1 else 'FAIL')

    print('-'*20)
    return True

############################################################################################
###############################   Section 2.2 - Conv2D  ######################################
############################################################################################

def conv2d_forward_correctness():
    '''
    lecture 9: pg 102
    lectur 10: pg 82
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    out_c = np.random.randint(5,15)
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(60,80)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    def random_normal_weight_init_fn(out_channel, in_channel, kernel_width, kernel_height):
        return np.random.normal(0, 1.0, (out_channel, in_channel, kernel_width, kernel_height))
    
    test_model = Conv2D(in_c, out_c, kernel, stride, random_normal_weight_init_fn, np.zeros)
    
    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride)
    torch_model.weight = nn.Parameter(torch.tensor(test_model.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.b))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()
    
    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    return scores_dict

def test_conv2d_forward():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a = conv2d_forward_correctness()[0]
        if a != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Forward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2D Forward Test: %d / %d' % (i + 1, n))
    return True

def conv2d_backward_correctness():
    '''
    lecture 9: pg 102
    lectur 10: pg 82
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0, 0, 0, 0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    out_c = np.random.randint(5,15)
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(60,80)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    def random_normal_weight_init_fn(out_channel, in_channel, kernel_width, kernel_height):
        return np.random.normal(0, 1.0, (out_channel, in_channel, kernel_width, kernel_height))
    
    test_model = Conv2D(in_c, out_c, kernel, stride, random_normal_weight_init_fn, np.zeros)
    
    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride)
    torch_model.weight = nn.Parameter(torch.tensor(test_model.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.b))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()
    
    b, c, w, h = torch_y.shape
    delta = np.random.randn(b, c, w, h)
    y1.backward(torch.tensor(delta))
    dy1 = x1.grad
    torch_dx = dy1.detach().numpy()
    torch_dW = torch_model.weight.grad.detach().numpy()
    torch_db = torch_model.bias.grad.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        dy2 = test_model.backward(delta)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_dx = dy2
    test_dW = test_model.dW
    test_db = test_model.db
    
    if not assertions(test_dx, torch_dx, 'type', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1
    
    if not assertions(test_dW, torch_dW, 'type', 'dW'): return scores_dict
    if not assertions(test_dW, torch_dW, 'shape', 'dW'): return scores_dict
    if not assertions(test_dW, torch_dW, 'closeness', 'dW'): return scores_dict
    scores_dict[2] = 1
    
    if not assertions(test_db, torch_db, 'type', 'db'): return scores_dict
    if not assertions(test_db, torch_db, 'shape', 'db'): return scores_dict
    if not assertions(test_db, torch_db, 'closeness', 'db'): return scores_dict
    scores_dict[3] = 1
    
    #############################################################################################
    ##############################    Compare Results   #########################################
    #############################################################################################
    
    return scores_dict

def test_conv2d_backward():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a, b, c, d = conv2d_backward_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1 or c != 1 or d != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2D Backward Test: %d / %d' % (i + 1, n))
    print('-'*20)
    return True


############################################################################################
####################################   Section 3.4    ######################################
############################################################################################


import mlp_scan as cnn_solution

def test_simple_scanning_mlp():
    data = np.loadtxt(os.path.join('autograder', 'hw2_autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
    cnn = cnn_solution.CNN_SimpleScanningMLP()
    weights = np.load(os.path.join('autograder', 'hw2_autograder', 'weights', 'mlp_weights_part_b.npy'), allow_pickle = True)
    cnn.init_weights(weights)

    expected_result = np.load(os.path.join('autograder', 'hw2_autograder', 'ref_result', 'res_b.npy'), allow_pickle = True)
    result = cnn(data)

    try:
        assert(type(result)==type(expected_result))
        assert(result.shape==expected_result.shape)
        assert(np.allclose(result,expected_result))

        print("Simple Scanning MLP:" + "PASS")
        print('-'*20)
        return True
    except Exception as e:
        print("Simple Scanning MLP:" + "FAIL")
        traceback.print_exc()
        print('-'*20)
        return False


def test_distributed_scanning_mlp():
    data = np.loadtxt(os.path.join('autograder', 'hw2_autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
    cnn = cnn_solution.CNN_DistributedScanningMLP()
    weights = np.load(os.path.join('autograder', 'hw2_autograder', 'weights', 'mlp_weights_part_c.npy'), allow_pickle = True)
    cnn.init_weights(weights)

    expected_result = np.load(os.path.join('autograder', 'hw2_autograder', 'ref_result', 'res_c.npy'), allow_pickle = True)
    result = cnn(data)

    try:
        assert(type(result)==type(expected_result))
        assert(result.shape==expected_result.shape)
        assert(np.allclose(result,expected_result))
        print("Distributed Scanning MLP:" + "PASS")
        print('-'*20)
        return True
    except Exception as e:
        print("Distributed Scanning MLP:" + "FAIL")
        print('-'*20)
        traceback.print_exc()
        return False

############################################################################################
#########   Section 3.5 - Build Your Own CNN Model    ######################################
############################################################################################
import hw2

# Default Weight Initialization for Conv1D and Linear
def conv1d_random_normal_weight_init(d0, d1, d2):
    return np.random.normal(0, 1.0, (d0, d1, d2))

def linear_random_normal_weight_init(d0, d1):
    return np.random.normal(0, 1.0, (d0, d1))

def zeros_bias_init(d):
    return np.zeros(d)

def get_cnn_model():
    input_width = 128
    input_channels = 24
    conv_weight_init_fn = conv1d_random_normal_weight_init
    linear_weight_init_fn = linear_random_normal_weight_init
    bias_init_fn = zeros_bias_init
    criterion = SoftmaxCrossEntropy()
    lr = 1e-3

    num_linear_neurons = 10
    out_channels = [56, 28, 14]
    kernel_sizes = [5, 6, 2]
    strides = [1, 2, 2]
    activations = [Tanh(), ReLU(), Sigmoid()]

    model = hw2.CNN(input_width, input_channels, out_channels, kernel_sizes, strides, num_linear_neurons,
                 activations, conv_weight_init_fn, bias_init_fn, linear_weight_init_fn,
                 criterion, lr)
    
    return model


class Flatten_(nn.Module):
    def __init__(self):
        super(Flatten_, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.in_channel = 128
        self.out_size = 10

        self.conv1 = nn.Conv1d(self.in_channel, 56, 5, 1)
        self.conv2 = nn.Conv1d(56, 28, 6, 2)
        self.conv3 = nn.Conv1d(28, 14, 2, 2)

        self.flatten = Flatten_()
        self.fc = nn.Linear(14 * 30, self.out_size)

    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


def cnn_model_correctness(idx):

    scores_dict = [0, 0, 0, 0]

    TOLERANCE = 1e-8
    
    '''
    Write assertions to check the weight dimensions for each layer and see whether they're initialized correctly
    '''

    submit_model = get_cnn_model()
    ref_model = CNN_model()
    for i in range(3):
        getattr(ref_model, 'conv{:d}'.format(i + 1)).weight = nn.Parameter(torch.tensor(submit_model.convolutional_layers[i].W))
        getattr(ref_model, 'conv{:d}'.format(i + 1)).bias = nn.Parameter(torch.tensor(submit_model.convolutional_layers[i].b))
    ref_model.fc.weight = nn.Parameter(torch.tensor(submit_model.linear_layer.W.T))
    ref_model.fc.bias = nn.Parameter(torch.tensor(submit_model.linear_layer.b))

    data = np.loadtxt(os.path.join('autograder', 'hw2_autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
    labels = np.load(os.path.join('autograder', 'hw2_autograder', 'data', 'labels.npy'), allow_pickle = True)

    #############################################################################################
    #########################    Get the correct results from Refrence   ########################
    #############################################################################################

    # Model architecture is hardcoded
    # Width: 128 -> Int((128 - 5) / 1) + 1 = 124
    #        124 -> Int((124 - 6) / 2) + 1 = 60
    #        60  -> Int((60 - 2) / 2) + 1  = 30

    x_torch = Variable(torch.from_numpy(data), requires_grad = True)
    labels_torch = torch.tensor([0.0]).long()

    y1 = ref_model(x_torch)
    b, w = y1.shape

    criterion = nn.CrossEntropyLoss()
    loss = criterion(y1, labels_torch)
    loss.backward()

    dx_ref = x_torch.grad.detach().numpy()
    dW_ref = ref_model.conv1.weight.grad.detach().numpy()
    db_ref = ref_model.conv1.bias.grad.detach().numpy()

    
    #############################################################################################
    ##########################    Get your forward results and compare ##########################
    #############################################################################################
    y2 = submit_model(data)
    assert y1.shape == y2.shape
    if not(y1.shape == y2.shape): print("FAILURE")

    forward_res = y2 - y1.detach().numpy()
    forward_res_norm = abs(forward_res).max()

    if forward_res_norm < TOLERANCE:
        scores_dict[0] = 1
    else:
        print("Fail to return correct forward values")
        assert False
        return scores_dict
    
    #############################################################################################
    ##################   Get your backward results and check the tensor shape ###################
    #############################################################################################
    dx = submit_model.backward(labels)
    dW = submit_model.convolutional_layers[0].dW
    db = submit_model.convolutional_layers[0].db

    assert dx.shape == data.shape
    assert dW_ref.shape == dW.shape
    assert db_ref.shape == db.shape

    #############################################################################################
    ################   Check your dx, dW and db with Reference #################
    #############################################################################################
    delta_res_norm = abs(dx - dx_ref).max()

    dW_res = dW - dW_ref
    dW_res_norm = abs(dW_res).max()

    db_res = db - db_ref
    db_res_norm = abs(db_res).max()


    if delta_res_norm < TOLERANCE:
        scores_dict[1] = 1
    
    if dW_res_norm < TOLERANCE:
        scores_dict[2] = 1
    
    if db_res_norm < TOLERANCE:
        scores_dict[3] = 1
    
    if min(scores_dict) != 1:
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dx')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values db')
        assert False
    return scores_dict

def test_conv1d_model():
    scores = []
    worker = min(mtp.cpu_count(), 4)
    p = mtp.Pool(worker)

    scores_dict = cnn_model_correctness(0)
    scores.append(scores_dict)
    if min(scores_dict) != 1:
        return False

    # scores = np.min(scores, axis = 0)
    a, b, c, d = np.array(scores).min(0)
    # print('Section 6 - CNN Complete Model | 15 points')

    print('Conv1D Model Forward:', 'PASS' if a == 1 else 'FAIL')
    print('Conv1D Model dX:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1D Model dW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1D Model db:', 'PASS' if d == 1 else 'FAIL')
    print('-'*20)
    return True


# TODO: add tests here with names and calling the functions. 
# 'autolab' is the name on autolab I think, but you probably won't need to worry about it.
# The test functions should return True or False.
tests = [
    {
        'name': '3.1 - MCQ 1 | 1 point',
        'autolab': 'MCQ 1',
        'handler': test_mcq_1,
        'value': 1,
    },
    {
        'name': '3.2 - MCQ 2 | 1 point',
        'autolab': 'MCQ 2',
        'handler': test_mcq_2,
        'value': 1,
    },
    {
        'name': '3.3 - MCQ 3 | 1 point',
        'autolab': 'MCQ 3',
        'handler': test_mcq_3,
        'value': 1,
    },
    {
        'name': '3.4 - MCQ 4 | 1 point',
        'autolab': 'MCQ 4',
        'handler': test_mcq_4,
        'value': 1,
    },
    {
        'name': '3.5 - MCQ 5 | 1 point',
        'autolab': 'MCQ 5',
        'handler': test_mcq_5,
        'value': 1,
    },
    {
        'name': '4.1 - Conv1D | 40 points',
        'autolab': 'Conv1D',
        'handler': test_cnn_correctness,
        'value': 40,
    },
    {
        'name': '4.2.1 - Conv2D (Forward) | 10 points',
        'autolab': 'Conv2D-forward',
        'handler': test_conv2d_forward,
        'value': 10,
    },
    {
        'name': '4.2.2 - Conv2D (Backward) | 10 points',
        'autolab': 'Conv2D-backward',
        'handler': test_conv2d_backward,
        'value': 10,
    },
    {
        'name': '5.1 - CNN as Simple Scanning MLP | 10 points',
        'autolab': 'CNN as Simple Scanning MLP',
        'handler': test_simple_scanning_mlp,
        'value': 10,
    },
    {
        'name': '5.2 - CNN as Distributed Scanning MLP | 10 points',
        'autolab': 'CNN as Distributed Scanning MLP',
        'handler': test_distributed_scanning_mlp,
        'value': 10,
    },
    {
        'name': '6 - Build a CNN Model | 15 points',
        'autolab': 'Build a CNN Model',
        'handler': test_conv1d_model,
        'value': 15,
    }


    
]
# tests.reverse()

# print('--------------------------------------------------')
# # Section 4

# # Section 4.1
# a, b, c, d = test_cnn_correctness()
# print('Section 4.1 - Convolutional Layer: Conv1D | 40 points')

# print('Section 4.1.1 - Forward | 15 points')
# print('Conv1D Forward:', 'PASS' if a == 1 else 'FAIL')

# print('Section 4.1.2 - Backward | 25 points')
# print('Conv1D dX:', 'PASS' if b == 1 else 'FAIL')
# print('Conv1D dW:', 'PASS' if c == 1 else 'FAIL')
# print('Conv1D db:', 'PASS' if d == 1 else 'FAIL')

# print('--------------------------------------------------')

# # Section 4.2
# cnv2f = test_conv2d_forward()
# print('Section 4.2 - Convolutional Layer: Conv2D | 20 points')

# print('Section 4.2.1 - Conv2D : Forward')
# print('Conv2D Forward:', 'PASS' if cnv2f == 1 else 'FAIL')

# cnv2b = test_conv2d_backward()
# print('Section 4.2.2 - Conv2D : Backward')
# print('Conv2D Backward:', 'PASS' if cnv2b == 1 else 'FAIL')

# print('--------------------------------------------------')


# # Section 5.1
# print('Section 5.1 - CNN as a Simple Scanning MLP | 10 points')
# b = test_simple_scanning_mlp()
# print("Scanning MLP:", "PASS" if b else "FAIL")

# print('--------------------------------------------------')

# # Section 5.2
# print('Section 5.2 - CNN as a Distributed Scanning MLP | 10 points')
# c = test_distributed_scanning_mlp()
# print("Distributed MLP:", "PASS" if c else "FAIL")    

# print('--------------------------------------------------')

# # Section 6
# a, b, c, d = test_conv1d_model()
# print('Section 6 - CNN Complete Model | 15 points')

# print('Conv1D Model Forward:', 'PASS' if a == 1 else 'FAIL')
# print('Conv1D Model dX:', 'PASS' if b == 1 else 'FAIL')
# print('Conv1D Model dW:', 'PASS' if c == 1 else 'FAIL')
# print('Conv1D Model db:', 'PASS' if d == 1 else 'FAIL')

if __name__=='__main__':
    # np.random.seed(2021)
    run_tests(tests)


