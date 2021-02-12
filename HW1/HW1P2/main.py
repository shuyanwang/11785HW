from learninghw1 import LearningHW1, ParamsHW1
from models.mlp4 import MLP4


def main():
    # params = ParamsHW1(15, 131072, max_epoch=51)
    # model = mlp2.MLP2(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()
    # del learner
    #
    # params = ParamsHW1(15, 65536, max_epoch=51)
    # model = mlp2.MLP2(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()

    # params = ParamsHW1(15, 65536, max_epoch=31)
    # model = mlp3.MLP3(params.K)
    # learner = LearningHW1(params, model)
    # learner.learn()
    # del learner

    params = ParamsHW1()
    model = MLP4(params.K)
    learner = LearningHW1(params, model)
    learner.learn()
    del learner


if __name__ == '__main__':
    main()
