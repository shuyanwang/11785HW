from learninghw1 import LearningHW1, ParamsHW1
from models import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
args = parser.parse_args()


# Fastest GPU is assigned as 0 -> Different from PCIE BUS ID


def main():
    params = ParamsHW1(B=8192, K=30, dropout=0.5, device='cuda:' + args.gpu_id)
    model = MLP17(params.K, params.dropout)
    learner = LearningHW1(params, model)
    learner.train()
    learner.test()


if __name__ == '__main__':
    main()
