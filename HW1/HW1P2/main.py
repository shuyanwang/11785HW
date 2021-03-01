from learninghw1 import LearningHW1, ParamsHW1
from models import *


def main():
    params = ParamsHW1(B=4096, K=30, dropout=0.5)
    model = MLP18(params.K, params.dropout)
    learner = LearningHW1(params, model)
    learner.load_model(36)
    # learner.train()
    learner.test()


if __name__ == '__main__':
    main()
