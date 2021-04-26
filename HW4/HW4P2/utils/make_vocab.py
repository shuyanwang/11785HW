import numpy as np
import os
import json

data_dir = 'c:/DLData/11785_data/HW4'


def make():
    train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
                      allow_pickle=True)  # (N,T_out), dtype=bytes
    valid_y = np.load(os.path.join(data_dir, 'dev_transcripts.npy'), allow_pickle=True)

    vocab = set()
    for article in train_y:
        for sequence in article:
            for character in sequence:
                vocab.add(chr(character))

    for article in valid_y:
        for sequence in article:
            for character in sequence:
                vocab.add(chr(character))

    vocab = list(vocab)
    vocab.sort()

    dictionary = {vocab[i]: (i + 1) for i in range(len(vocab))}  # 1->27; <eol> is 0

    with open('vocab.json', 'w') as f:
        json.dump(dictionary, f)


def parse():
    with open('vocab.json') as f:
        dictionary = json.load(f)

        train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
                          allow_pickle=True)  # (N,T_out), dtype=bytes

        train_y_np = []

        for article in train_y:
            for sequence in article:
                sequence_np = np.zeros((len(sequence) + 1), dtype=int)
                for i, character in enumerate(sequence):
                    sequence_np[i] = dictionary[chr(character)]

                train_y_np.append(sequence_np)

        train_y_np = np.asarray(train_y_np, dtype=object)

        np.save(os.path.join(data_dir, 'train_labels.npy'), train_y_np)

        valid_y = np.load(os.path.join(data_dir, 'dev_transcripts.npy'), allow_pickle=True)

        valid_y_np = []

        for article in valid_y:
            for sequence in article:
                sequence_np = np.zeros((len(sequence) + 1), dtype=int)
                for i, character in enumerate(sequence):
                    sequence_np[i] = dictionary[chr(character)]

                valid_y_np.append(sequence_np)

        valid_y_np = np.asarray(valid_y_np, dtype=object)

        np.save(os.path.join(data_dir, 'dev_labels.npy'), valid_y_np)


if __name__ == '__main__':
    # make()
    # parse()
    pass
