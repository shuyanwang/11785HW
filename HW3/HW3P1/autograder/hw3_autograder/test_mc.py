# DO NOT EDIT this file. It is set up in such a way that if you make any edits,
# the test cases may change resulting in a broken local autograder.

# Imports
import sys, os, pdb
from test import Test

# Append paths and run
sys.path.append("hw3")
import mc


############################################################################################
################################   Section 1 - MCQ    ######################################
############################################################################################


class MCQTest(Test):
    def __init__(self):
        pass

    def test_mutiple_choice(self):
        scores = [0, 0, 0, 0]

        ref = ["b", "b", "b", "a"]
        ans_1 = mc.question_1()
        ans_2 = mc.question_2()
        ans_3 = mc.question_3()
        ans_4 = mc.question_4()
        ans = [ans_1, ans_2, ans_3, ans_4]

        for i in range(len(ref)):
            if ref[i] == ans[i]:
                scores[i] = 1

        return scores

    def run_test(self):
        # Test MCQs
        self.print_name("Section 1 - Multiple Choice Questions")
        a, b, c, d = self.test_mutiple_choice()
        all_correct = a and b and c and d
        self.print_outcome("Multiple Choice Questions", all_correct)
        if all_correct == False:
            self.print_failure("Multiple Choice Questions")
            return False

        return True
