import numpy as np


def main():
    a = np.load('../dataset/vocab.npy')  # [str,...]
    b = np.load('../dataset/wiki.train.npy', allow_pickle=True)  # [[int,...],...]
    t = 1


if __name__ == '__main__':
    main()
