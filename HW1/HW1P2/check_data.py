import numpy as np
import os

data_dir = 'E:/11785_data/HW1'

dev_x = np.load(os.path.join(data_dir, 'dev.npy'), allow_pickle=True)

t = dev_x[5]

t = 1
