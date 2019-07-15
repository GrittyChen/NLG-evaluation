#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

from Metrics.bleu.bleu import Bleu
from Metrics.rouge.rouge import Rouge
from Metrics.meteor.meteor import Meteor
from Metrics.cider.cider import Cider


class Evaluate(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE-L"),
            (Cider(), "CIDEr")
        ]

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))
        else:
            return data

    def score(self, refs, hypos):
        final_scores = {}
        for scorer, metric in self.scorers:
            score, scores = scorer.compute_score(refs, hypos)
            if type(metric) is list:
                for m, s in zip(metric, score):
                    final_scores[m] = s
            else:
                final_scores[metric] = score

        return final_scores

    def evaluate(self, get_scores=True, live=False, **kwargs):
        if live:
            tmp_refs = kwargs.pop("refs", {})
            tmp_hypos = kwargs.pop("hypos", {})

            refs = {}
            hypos = {}
            idx = 0
            for k, v in tmp_hypos.items():
                hypos[idx] = [v]
                refs[idx] = tmp_refs[k]
                idx += 1
        else:
            refs_file = kwargs.pop("refs", "")
            hypos_file = kwargs.pop("hypos", "")

            with open(refs_file) as fd:
                refs = fd.readlines()
                refs = {ids: line.strip().split("\t") for ids, line in enumerate(refs)}

            with open(hypos_file) as fd:
                hypos = fd.readlines()
                hypos = {ids: [line.strip()] for ids, line in enumerate(hypos)}

        final_scores = self.score(refs, hypos)

        print("BLEU-1:\t", final_scores["BLEU-1"])
        print("BLEU-2:\t", final_scores["BLEU-2"])
        print("BLEU-3:\t", final_scores["BLEU-3"])
        print("BLEU-4:\t", final_scores["BLEU-4"])
        print("METEOR:\t", final_scores["METEOR"])
        print("ROUGE-L:\t", final_scores["ROUGE-L"])
        print("CIDEr:\t", final_scores["CIDEr"])

        if get_scores:
            return final_scores


if __name__ == "__main__":
    _hypos = sys.argv[1]
    _refs = sys.argv[2]

    obj = Evaluate()

    res = obj.evaluate(hypos=_hypos, refs=_refs)

