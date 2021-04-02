import torch
from typing import Dict, List
import numpy as np


def greedy_search(symbols, logits):
    """

    :param symbols:
    :param logits: (T,42)
    :return:
    """
    path = []
    end_with_blank = False
    p = 0
    for t in range(logits.shape[0]):
        p += torch.max(logits[t, :])
        index = torch.argmax(logits[t, :])
        if index != 0:
            if end_with_blank:
                # path.append(symbols[0])
                path.append(symbols[index])
                end_with_blank = False
            else:
                if len(path) == 0 or path[-1] != symbols[index]:
                    path.append(symbols[index])
        else:
            end_with_blank = True

    return ''.join(path), p


class BeamSearchClass:
    def __init__(self, symbols, k):
        self.symbols = symbols
        self.k = k

    def decode(self, logits):
        self.y_probs = logits
        self.paths_blank: List[str] = ['']
        self.paths_blank_score: Dict[str:np.ndarray] = {'': logits[0, 0]}

        self.paths_symbol: List[str] = [c for c in self.symbols]
        self.paths_symbol_score: Dict[str:np.ndarray] = {}
        for i, c in enumerate(self.symbols):
            self.paths_symbol_score[c] = logits[0, i]

        for t in range(1, self.y_probs.shape[0]):
            self.prune()
            updated_paths_symbol, updated_paths_symbol_score = self.extend_with_symbol(t)
            updated_paths_blank, updated_paths_blank_score = self.extend_with_blank(t)
            self.paths_blank = updated_paths_blank
            self.paths_symbol = updated_paths_symbol
            self.paths_blank_score = updated_paths_blank_score
            self.paths_symbol_score = updated_paths_symbol_score

        return self.merge()

    def extend_with_symbol(self, t):
        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        for path in self.paths_blank:
            for i, c in enumerate(self.symbols):
                new_path = path + c
                updated_paths_symbol.append(new_path)
                updated_paths_symbol_score[new_path] = self.paths_blank_score[path] + self.y_probs[
                    t, i]

        for path in self.paths_symbol:
            for i, c in enumerate(self.symbols):
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_symbol_score:
                    updated_paths_symbol_score[new_path] += self.paths_symbol_score[path] + \
                                                            self.y_probs[t, i]
                else:
                    updated_paths_symbol_score[new_path] = self.paths_symbol_score[path] + \
                                                           self.y_probs[t, i]
                    updated_paths_symbol.append(new_path)

        return updated_paths_symbol, updated_paths_symbol_score

    def extend_with_blank(self, t):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        for path in self.paths_blank:
            updated_paths_blank.append(path)
            updated_paths_blank_score[path] = self.paths_blank_score[path] + self.y_probs[t, 0]

        for path in self.paths_symbol:
            if path in updated_paths_blank:
                updated_paths_blank_score[path] += self.paths_symbol_score[path] + self.y_probs[
                    t, 0]
            else:
                updated_paths_blank_score[path] = self.paths_symbol_score[path] + self.y_probs[
                    t, 0]
                updated_paths_blank.append(path)

        return updated_paths_blank, updated_paths_blank_score

    def prune(self):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        scores = []

        for score in self.paths_blank_score.values():
            scores.append(score)

        for score in self.paths_symbol_score.values():
            scores.append(score)

        scores.sort()

        cutoff = scores[-1] if len(scores) < self.k else scores[- self.k]

        for path in self.paths_blank:
            if self.paths_blank_score[path] >= cutoff:
                updated_paths_blank.append(path)
                updated_paths_blank_score[path] = self.paths_blank_score[path]

        for path in self.paths_symbol:
            if self.paths_symbol_score[path] >= cutoff:
                updated_paths_symbol.append(path)
                updated_paths_symbol_score[path] = self.paths_symbol_score[path]

        self.paths_symbol_score = updated_paths_symbol_score
        self.paths_symbol = updated_paths_symbol
        self.paths_blank_score = updated_paths_blank_score
        self.paths_blank = updated_paths_blank

    def merge(self):
        paths = self.paths_blank
        scores = self.paths_blank_score

        for path in self.paths_symbol:
            if path in paths:
                scores[path] += self.paths_symbol_score[path]
            else:
                paths.append(path)
                scores[path] = self.paths_symbol_score[path]

        max_path = paths[0]
        max_score = scores[max_path]
        for path in scores:
            if scores[path] > max_score:
                max_path = path
                max_score = scores[path]

        return max_path, scores
