import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        extSymbols = []
        skipConnect = []

        for i, item in enumerate(target):
            extSymbols.append(self.BLANK)
            skipConnect.append(False)

            extSymbols.append(item)
            skipConnect.append(i > 0 and target[i] != target[i - 1])

        extSymbols.append(self.BLANK)
        skipConnect.append(False)

        return np.asarray(extSymbols), np.asarray(skipConnect)

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        alpha[0, 0] = logits[0, extSymbols[0]]
        alpha[0, 1] = logits[0, extSymbols[1]]

        # for t in range(1, T):
        #     alpha[t, 0] = alpha[t - 1, 0] + logits[t, extSymbols[0]]
        #     for i in range(1, S):
        #         if skipConnect[i]:
        #             alpha[t, i] = np.log(
        #                     np.exp(alpha[t - 1, i - 1]) + np.exp(alpha[t - 1, i]) + np.exp(
        #                             alpha[t - 1, i - 2])) + logits[t, extSymbols[i]]
        #         else:
        #             alpha[t, i] = np.log(np.exp(alpha[t - 1, i - 1]) + np.exp(alpha[t - 1,
        #             i])) + \
        #                           logits[t, extSymbols[i]]
        #
        # return alpha

        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] * logits[t, extSymbols[0]]
            for i in range(1, S):
                if skipConnect[i]:
                    alpha[t, i] = alpha[t - 1, i - 1] + alpha[t - 1, i] + alpha[t - 1, i - 2]
                else:
                    alpha[t, i] = alpha[t - 1, i - 1] + alpha[t - 1, i]
                alpha[t, i] *= logits[t, extSymbols[i]]

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        beta[-1, -1] = 1
        beta[-1, -2] = 1

        for t in range(T - 2, -1, -1):
            beta[t, -1] = beta[t + 1, -1] * logits[t + 1, extSymbols[-1]]
            for i in range(S - 2, -1, -1):
                if i + 2 < S - 1 and skipConnect[i + 2]:
                    beta[t, i] = beta[t + 1, i] * logits[t + 1, extSymbols[i]] + beta[
                        t + 1, i + 1] * logits[t + 1, extSymbols[i + 1]] + beta[t + 1, i + 2] * \
                                 logits[t + 1, extSymbols[i + 2]]
                else:
                    beta[t, i] = beta[t + 1, i] * logits[t + 1, extSymbols[i]] + beta[
                        t + 1, i + 1] * logits[t + 1, extSymbols[i + 1]]

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """

        # T = alpha.shape[0]
        # S = alpha.shape[1]
        # gamma = np.zeros_like(alpha)
        #
        # for t in range(T):
        #     for i in range(S):
        #         gamma[t, i] = alpha[t, i] * beta[t, i]
        #     sum_gamma_t = np.sum(gamma[t])
        #
        #     gamma[t] /= sum_gamma_t

        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1).reshape((-1, 1))

        return gamma
