import numpy as np
import sys, os, pdb
import pickle
from test import Test

sys.path.append("mytorch")
from ctc_loss import *
from ctc import *


data_path = os.path.join("autograder", "hw3_autograder", "data")
ref_data_path = os.path.join("autograder", "hw3_autograder", "data", "ctc_ref_data")


#################################################################################################
################################   Section 4 - CTC Loss    ######################################
#################################################################################################


class CTCTest(Test):
    def __init__(self):
        pass

    def test_ctc_extend_seq(self):
        # Get curr data
        probs = np.load(os.path.join(data_path, "X.npy"))
        targets = np.load(os.path.join(data_path, "Y.npy"))
        input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
        out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

        CTC_user = CTC(BLANK=0)

        f_ref_S_ext = open(os.path.join(ref_data_path, "ref_S_ext.pkl"), "rb")
        f_ref_Skip_Connect = open(
            os.path.join(ref_data_path, "ref_Skip_Connect.pkl"), "rb"
        )

        ref_S_ext_ls = pickle.load(f_ref_S_ext)
        ref_Skip_Connect_ls = pickle.load(f_ref_Skip_Connect)

        _, B, _ = probs.shape
        for b in range(B):
            target = targets[b, : out_lens[b]]

            user_S_ext, user_Skip_Connect = CTC_user.targetWithBlank(target)
            user_S_ext, user_Skip_Connect = (
                np.array(user_S_ext),
                np.array(user_Skip_Connect),
            )

            ref_S_ext = ref_S_ext_ls[b]
            ref_Skip_Connect = ref_Skip_Connect_ls[b]

            if not self.assertions(user_S_ext, ref_S_ext, "type", "extSymbols"):
                return False
            if not self.assertions(user_S_ext, ref_S_ext, "shape", "extSymbols"):
                return False
            if not self.assertions(user_S_ext, ref_S_ext, "closeness", "extSymbols"):
                return False

            if not self.assertions(
                user_Skip_Connect, ref_Skip_Connect, "type", "Skip_Connect"
            ):
                return False
            if not self.assertions(
                user_Skip_Connect, ref_Skip_Connect, "shape", "Skip_Connect"
            ):
                return False
            if not self.assertions(
                user_Skip_Connect, ref_Skip_Connect, "closeness", "Skip_Connect"
            ):
                return False

        f_ref_S_ext.close()
        f_ref_Skip_Connect.close()

        return True

    def test_ctc_posterior_prob(self):
        # Get curr data
        probs = np.load(os.path.join(data_path, "X.npy"))
        targets = np.load(os.path.join(data_path, "Y.npy"))
        input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
        out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

        CTC_user = CTC(BLANK=0)

        f_ref_alpha = open(os.path.join(ref_data_path, "ref_alpha.pkl"), "rb")
        f_ref_beta = open(os.path.join(ref_data_path, "ref_beta.pkl"), "rb")
        f_ref_gamma = open(os.path.join(ref_data_path, "ref_gamma.pkl"), "rb")

        ref_alpha_ls = pickle.load(f_ref_alpha)
        ref_beta_ls = pickle.load(f_ref_beta)
        ref_gamma_ls = pickle.load(f_ref_gamma)

        _, B, _ = probs.shape
        for b in range(B):
            logit = probs[: input_lens[b], b]
            target = targets[b, : out_lens[b]]

            user_S_ext, user_Skip_Connect = CTC_user.targetWithBlank(target)
            user_alpha = CTC_user.forwardProb(logit, user_S_ext, user_Skip_Connect)
            user_beta = CTC_user.backwardProb(logit, user_S_ext, user_Skip_Connect)
            user_gamma = CTC_user.postProb(user_alpha, user_beta)

            ref_alpha = ref_alpha_ls[b]
            ref_beta = ref_beta_ls[b]
            ref_gamma = ref_gamma_ls[b]

            if not self.assertions(user_alpha, ref_alpha, "type", "alpha"):
                return False
            if not self.assertions(user_alpha, ref_alpha, "shape", "alpha"):
                return False
            if not self.assertions(user_alpha, ref_alpha, "closeness", "alpha"):
                return False

            if not self.assertions(user_beta, ref_beta, "type", "beta"):
                return False
            if not self.assertions(user_beta, ref_beta, "shape", "beta"):
                return False
            if not self.assertions(user_beta, ref_beta, "closeness", "beta"):
                return False

            if not self.assertions(user_gamma, ref_gamma, "type", "gamma"):
                return False
            if not self.assertions(user_gamma, ref_gamma, "shape", "gamma"):
                return False
            if not self.assertions(user_gamma, ref_gamma, "closeness", "gamma"):
                return False

        f_ref_alpha.close()
        f_ref_beta.close()
        f_ref_gamma.close()

        return True

    def test_ctc_forward(self):
        # Get curr data
        probs = np.load(os.path.join(data_path, "X.npy"))
        targets = np.load(os.path.join(data_path, "Y.npy"))
        input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
        out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

        CTC_user = CTCLoss(BLANK=0)
        user_loss = CTC_user(probs, targets, input_lens, out_lens)

        ref_loss = np.load(os.path.join(ref_data_path, "ref_loss.npy"))

        if not self.assertions(user_loss, ref_loss, "closeness", "forward"):
            return False

        return True

    def test_ctc_backward(self):
        # Get curr data
        probs = np.load(os.path.join(data_path, "X.npy"))
        targets = np.load(os.path.join(data_path, "Y.npy"))
        input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
        out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

        CTC_user = CTCLoss(BLANK=0)
        user_loss = CTC_user(probs, targets, input_lens, out_lens)
        user_dy = CTC_user.backward()

        ref_dy = np.load(os.path.join(ref_data_path, "ref_dy.npy"))

        if not self.assertions(user_dy, ref_dy, "type", "backward"):
            return False
        if not self.assertions(user_dy, ref_dy, "closeness", "backward"):
            return False

        return True

    def run_test(self):
        # Test Extend Sequence with Blank
        self.print_name("Section 4 - Extend Sequence with Blank")
        extend_outcome = self.test_ctc_extend_seq()
        self.print_outcome("Extend Sequence with Blank", extend_outcome)
        if extend_outcome == False:
            self.print_failure("Extend Sequence with Blank")
            return False

        # Test Posterior Probability
        self.print_name("Section 4 - Posterior Probability")
        posterior_outcome = self.test_ctc_posterior_prob()
        self.print_outcome("Posterior Probability", posterior_outcome)
        if posterior_outcome == False:
            self.print_failure("Posterior Probability")
            return False

        # Test forward
        self.print_name("Section 4.1 - CTC Forward")
        forward_outcome = self.test_ctc_forward()
        self.print_outcome("CTC Forward", forward_outcome)
        if forward_outcome == False:
            self.print_failure("CTC Forward")
            return False

        # Test Backward
        self.print_name("Section 4.2 - CTC Backward")
        backward_outcome = self.test_ctc_backward()
        self.print_outcome("CTC backward", backward_outcome)
        if backward_outcome == False:
            self.print_failure("CTC Backward")
            return False

        return True
