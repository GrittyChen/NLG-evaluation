# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _lcs(string, sub):
    """
    Computes longest common subsequence (LCS) for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the LCS between the two strings
    """

    if len(string) < len(sub):
        sub, string = string, sub

    str_len, sub_len = len(string), len(sub)
    lengths = [[0 for _ in range(sub_len + 1)] for _ in range(str_len + 1)]

    for j in range(1, sub_len + 1):
        for i in range(1, str_len + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[str_len][sub_len]


class Rouge(object):
    """
    Class for computing ROUGE-L score for a set of 
    candidate sentences for the MS COCO test set
    """

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """

        assert(len(candidate) == 1)
        assert(len(refs) > 0)

        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split()

        for reference in refs:
            # split into tokens
            token_r = reference.split()
            # compute the longest common subsequence
            lcs = _lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / \
                float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0

        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and 
        candidate sentences for the dataset.
        :param gts: dict : ground_truth
        :param res: dict : results of predict
        :returns: average_score: float (mean ROUGE-L score)
        """

        score = []

        for idx in sorted(gts.keys()):
            hypo = res[idx]
            ref = gts[idx]
            score.append(self.calc_score(hypo, ref))

            # Sanity check
            assert(isinstance(hypo, list))
            assert(isinstance(ref, list))
            assert(len(hypo) == 1)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))

        # convert to percentage
        return 100 * average_score, np.array(score)

    @staticmethod
    def method():
        return "ROUGE-L"
