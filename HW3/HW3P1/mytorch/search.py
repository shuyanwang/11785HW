from typing import Dict, List

import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)

    # return (forward_path, forward_prob)
    path = []
    end_with_blank = False
    p = 1
    for t in range(y_probs.shape[1]):
        p *= np.max(y_probs[:, t, 0])
        index = np.argmax(y_probs[:, t, 0])
        if index != 0:
            if end_with_blank:
                path.append(SymbolSets[index - 1])
                end_with_blank = False
            else:
                if len(path) == 0 or path[-1] != SymbolSets[index - 1]:
                    path.append(SymbolSets[index - 1])
        else:
            end_with_blank = True

    return ''.join(path), p


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """

    bs = BeamSearchClass(SymbolSets, y_probs, BeamWidth)
    return bs()


class BeamSearchClass:
    def __init__(self, SymbolSets, y_probs, BeamWidth):
        self.symbols = SymbolSets
        self.y_probs = y_probs
        self.k = BeamWidth

        self.paths_blank: List[str] = ['']
        self.paths_blank_score: Dict[str:np.ndarray] = {'': y_probs[0, 0, 0]}

        self.paths_symbol: List[str] = [c for c in self.symbols]
        self.paths_symbol_score: Dict[str:np.ndarray] = {}
        for i, c in enumerate(SymbolSets):
            self.paths_symbol_score[c] = y_probs[i + 1, 0, 0]

    def __call__(self):
        for t in range(1, self.y_probs.shape[1]):
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
                updated_paths_symbol_score[new_path] = self.paths_blank_score[path] * self.y_probs[
                    i + 1, t, 0]

        for path in self.paths_symbol:
            for i, c in enumerate(self.symbols):
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_symbol_score:
                    updated_paths_symbol_score[new_path] += self.paths_symbol_score[path] * \
                                                            self.y_probs[i + 1, t, 0]
                else:
                    updated_paths_symbol_score[new_path] = self.paths_symbol_score[path] * \
                                                           self.y_probs[i + 1, t, 0]
                    updated_paths_symbol.append(new_path)

        return updated_paths_symbol, updated_paths_symbol_score

    def extend_with_blank(self, t):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        for path in self.paths_blank:
            updated_paths_blank.append(path)
            updated_paths_blank_score[path] = self.paths_blank_score[path] * self.y_probs[0, t, 0]

        for path in self.paths_symbol:
            if path in updated_paths_blank:
                updated_paths_blank_score[path] += self.paths_symbol_score[path] * self.y_probs[
                    0, t, 0]
            else:
                updated_paths_blank_score[path] = self.paths_symbol_score[path] * self.y_probs[
                    0, t, 0]
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
