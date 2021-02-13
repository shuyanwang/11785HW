from learninghw1 import LearningHW1, ParamsHW1
from models.model import *


def main():
    # params = ParamsHW1(15, 131072, max_epoch=51)
    # model = MLP2(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()
    # del learner
    #
    # params = ParamsHW1(15, 65536, max_epoch=51)
    # model = MLP2(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()

    # params = ParamsHW1(15, 65536, max_epoch=31)
    # model = MLP3(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()
    # del learner

    # params = ParamsHW1(lr=1e-3)
    # model = MLP4(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()
    #
    # del learner
    #
    # params = ParamsHW1()
    # model = MLP4(params.K)
    # learner = LearningHW1(params, model)
    # learner.load_model(epoch=5)
    # learner.learn()
    # del learner

    # change lr to 1e-3

    # params = ParamsHW1()
    # model = MLP5(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()

    # params = ParamsHW1(lr=1e-3)
    # model = MLP5(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()

    # params = ParamsHW1(B=32768, is_double=True, max_epoch=81, lr=1e-3)
    # model = MLP5(params.K)
    # learner = LearningHW1(params, model)
    # # learner.learn()
    # learner.load_model(65)
    # learner.test()

    # params = ParamsHW1(B=32768, is_double=True, max_epoch=81)
    # model = MLP5(params.K)
    # learner = LearningHW1(params, model)
    # learner.load_model(epoch=50)
    # learner.learn()

    params = ParamsHW1(B=32768, is_double=False, max_epoch=101, lr=1e-3)
    model = MLP5(params.K)
    learner = LearningHW1(params, model)
    learner.train()
    learner.test()

    # DO NOT USE DOUBLE, ALSO DO NOT PUSH B TO EXTREME


if __name__ == '__main__':
    main()
