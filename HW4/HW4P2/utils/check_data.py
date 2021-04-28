import numpy as np
import os

#
data_dir = 'c:/DLData/11785_data/HW4'
#
# train_x = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)  # (N,T_in,40)
#
# a = train_x[0]
#
# train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
#                   allow_pickle=True, encoding='bytes')  # (N,T_out), dtype=bytes
#
# for article in train_y:
#     for sequence in article:
#         t = 1
#
# b = train_y[0][0]
# #
# t = 1

train_x = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)  # (N,T_in,40)
train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
# (N,T_OUT) ...,<EOS>

t = 1
