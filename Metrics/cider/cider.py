# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Metrics.cider.cider_scorer import CiderScorer


class Cider(object):
    """
    Main Class to compute the CIDEr metric
    """

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  res: dict with value <tokenized candidate sentence>
        :param  gts: dict with value <tokenized reference sentence>
        :return: cider (float): computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for idx in sorted(gts.keys()):
            hypo = res[idx]
            ref = gts[idx]

            # Sanity check.
            assert(isinstance(hypo, list))
            assert(isinstance(ref, list))
            assert(len(hypo) == 1)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        return cider_scorer.compute_score()

    @staticmethod
    def method():
        return "CIDEr"
