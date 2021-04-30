import numpy as np
import os
import json

data_dir = 'c:/DLData/11785_data/HW4/hw4p2_simple'


def make_x():
    train_x = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)  # (N,T_in,40)

    for article in train_x:
        for sequence in article:
            t = 0


def make_labels():
    train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
                      allow_pickle=True)  # (N,T_out), dtype=bytes
    valid_y = np.load(os.path.join(data_dir, 'dev_transcripts.npy'), allow_pickle=True)

    vocab = set()
    for sentence in train_y:
        for word in sentence:
            for character in word:
                vocab.add(character)

    for sentence in valid_y:
        for word in sentence:
            for character in word:
                vocab.add(character)

    vocab = list(vocab)
    vocab.sort()

    dictionary = {vocab[i]: (i + 1) for i in range(len(vocab))}  # 1->27; <eol> is 0

    dictionary[" "] = len(dictionary) + 1

    with open('vocab_toy.json', 'w') as f:
        json.dump(dictionary, f)


def parse():
    with open('vocab_toy.json') as f:
        dictionary = json.load(f)

        train_y = np.load(os.path.join(data_dir, 'train_transcripts.npy'),
                          allow_pickle=True)  # (N,T_out), dtype=bytes

        train_y_np = []

        for sentence in train_y:
            sentence_np = []
            for word in sentence:
                for i, character in enumerate(word):
                    sentence_np.append(dictionary[character])
                sentence_np.append(dictionary[" "])
            sentence_np.append(0)

            train_y_np.append(np.asarray(sentence_np, dtype=int))

        train_y_np = np.asarray(train_y_np, dtype=object)

        np.save(os.path.join(data_dir, 'train_labels.npy'), train_y_np)

        valid_y = np.load(os.path.join(data_dir, 'dev_transcripts.npy'), allow_pickle=True)

        valid_y_np = []

        for sentence in valid_y:
            sentence_np = []
            for word in sentence:
                for i, character in enumerate(word):
                    sentence_np.append(dictionary[character])
                sentence_np.append(dictionary[" "])
            sentence_np.append(0)

            valid_y_np.append(np.asarray(sentence_np, dtype=int))

        valid_y_np = np.asarray(valid_y_np, dtype=object)

        np.save(os.path.join(data_dir, 'dev_labels.npy'), valid_y_np)


if __name__ == '__main__':
    # make_x()
    make_labels()
    parse()
    pass
