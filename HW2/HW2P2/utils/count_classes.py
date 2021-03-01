import numpy as np
import os


def main():
    data = np.load(os.path.join('E:/11785_data/HW1', 'train_labels.npy'), allow_pickle=True)
    data = np.concatenate(data).astype(np.long)
    y_count = np.bincount(data)

    np.save('E:/11785_data/HW1/train_labels_count.npy', y_count)


if __name__ == '__main__':
    main()
