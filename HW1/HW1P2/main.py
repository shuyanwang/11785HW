from learninghw1 import LearningHW1, ParamsHW1
from models import mlp2


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

    for k in range(10, 20):
        params = ParamsHW1(k, 65536, max_epoch=21)
        model = mlp2.MLP2(params.K)
        learner = LearningHW1(params, model)
        learner.learn()
        del learner


if __name__ == '__main__':
    main()
