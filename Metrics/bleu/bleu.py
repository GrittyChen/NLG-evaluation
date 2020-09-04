# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Metrics.bleu.bleu_scorer import BleuScorer


class Bleu(object):

    def __init__(self, n=4):
        # default compute BLEU score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):
        bleu_scorer = BleuScorer(n=self._n)

        for idx in sorted(gts.keys()):
            hypo = res[idx]
            ref = gts[idx]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)

        return score, scores

    @staticmethod
    def method():
        return "BLEU"
