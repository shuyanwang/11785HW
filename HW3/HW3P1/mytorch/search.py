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
    # Follow the pseudocode from lecture to complete beam search :-)

    # return (bestPath, mergedPathScores)
    raise NotImplementedError
