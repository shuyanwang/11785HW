import numpy as np
import os

data_dir = '/home/zongyuez/dldata/HW3'

dev_x = np.load(os.path.join(data_dir, 'dev.npy'), allow_pickle=True)

t = dev_x[5]
train_x = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)
train_y = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
# train_y: [1, 41]
for i in range(train_y.shape[0]):
    x_o = train_x[i]
    y_0 = np.asarray(train_y[i])
    print(np.max(y_0))

t = 1
