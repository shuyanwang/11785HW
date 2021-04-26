import numpy as np
import os

data_dir = 'c:/DLData/11785_data/HW4'

train_x = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)  # (N,T_in,40)

a = train_x[0]

train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
                  allow_pickle=True)  # (N,T_out), dtype=bytes

b = train_y[0][0]

t = 1
