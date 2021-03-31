import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        self.gammas = []
        for b in range(B):
            ctc = CTC(self.BLANK)
            logits_b = logits[0:input_lengths[b], b]
            extSymbols, skipConnect = ctc.targetWithBlank(target[b, 0:target_lengths[b]])
            alpha = ctc.forwardProb(logits_b, extSymbols, skipConnect)
            beta = ctc.backwardProb(logits_b, extSymbols, skipConnect)
            gamma = ctc.postProb(alpha, beta)
            for r in range(gamma.shape[1]):
                totalLoss[b] -= np.sum(gamma[0:, r] * np.log(logits_b[:, extSymbols[r]]))

            self.gammas.append(gamma)

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: scalar
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            gamma = self.gammas[b]
            ctc = CTC(self.BLANK)
            logits_b = self.logits[0:self.input_lengths[b], b]
            extSymbols, _ = ctc.targetWithBlank(self.target[b, 0:self.target_lengths[b]])
            for r in range(gamma.shape[1]):
                dY[0:self.input_lengths[b], b, extSymbols[r]] -= gamma[:, r] / logits_b[:,
                                                                               extSymbols[r]]
            # <---------------------------------------------

        return dY
