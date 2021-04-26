import json
import sys
import traceback
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from test import *

sys.path.append('mytorch')
# from conv import *
from pool import *


############################################################################################
###############################   Section 3 - MaxPool  #####################################
############################################################################################

def max_pool_correctness():
    '''
    lecture 10: pg 42, pg 164, pg 165
    Max Pooling Layer
    '''
    scores_dict = [0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    kernel = np.random.randint(3, 7)
    stride = np.random.randint(3, 5)
    width = np.random.randint(50, 100)
    in_c = np.random.randint(5, 15)
    batch = np.random.randint(1, 4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    torch_max_pool = nn.MaxPool2d(kernel, stride, return_indices=True)
    torch_max_unpool = nn.MaxUnpool2d(kernel, stride)

    test_model = MaxPoolLayer(kernel, stride)

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = torch.tensor(x)
    y1, indices = torch_max_pool(x1)
    torch_y = y1.detach().numpy()
    x1p = torch_max_unpool(y1, indices, output_size=x1.shape)
    torch_xp = x1p.detach().numpy()

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
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1

    return scores_dict


def test_max_pool():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b = max_pool_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed MaxPool Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print('Failed MaxPool Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MaxPool Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
###############################   Section 4 - MeanPool  ####################################
############################################################################################

def mean_pool_correctness():
    '''
    lecture 10: pg 44, pg 168, pg 169
    Mean Pooling Layer
    '''
    scores_dict = [0, 0]

    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    kernel = np.random.randint(3, 7)
    stride = np.random.randint(3, 5)
    width = np.random.randint(50, 100)
    in_c = np.random.randint(5, 15)
    batch = np.random.randint(1, 4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    torch_model = nn.functional.avg_pool2d

    test_model = MeanPoolLayer(kernel, stride)

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1, kernel, stride)
    torch_y = y1.detach().numpy()
    y1.backward(y1)
    x1p = x1.grad
    torch_xp = x1p.detach().numpy()

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
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1

    return scores_dict


def test_mean_pool():
    n = 3
    np.random.seed(11785)
    for i in range(n):
        a, b = mean_pool_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MeanPool Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
#################################### DO NOT EDIT ###########################################
############################################################################################

if __name__ == '__main__':

    tests = [

        {
            'name': 'Section 3 - MaxPool',
            'autolab': 'MaxPool',
            'handler': test_max_pool,
            'value': 5,
        },
        {
            'name': 'Section 4 - MeanPool',
            'autolab': 'MeanPool',
            'handler': test_mean_pool,
            'value': 5,
        },
    ]

    scores = {}
    for t in tests:
        print_name(t['name'])
        res = t['handler']()
        print_outcome(t['autolab'], res)
        scores[t['autolab']] = t['value'] if res else 0

    print(json.dumps({'scores': scores}))
