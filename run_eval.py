#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import collections

from Metrics.bleu.bleu import Bleu
from Metrics.rouge.rouge import Rouge
from Metrics.meteor.meteor import Meteor
from Metrics.cider.cider import Cider


def parse_args():
    parser = argparse.ArgumentParser(
        description="automatic evaluation for NLG systems",
        usage="run_eval.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--hypos", type=str, required=True,
                        help="Path of hypothesis file")
    parser.add_argument("--refs", type=str, required=True, nargs="+",
                        help="Path of reference file")

    # metrics
    parser.add_argument("-n", "--ngram", type=int, default=4,
                        help="calculate BLEU-n score")
    parser.add_argument("-lc", "--lowercase", action="store_true",
                        help="evaluation in lowercase mode")
    parser.add_argument("-nB", "--no_BLEU", action="store_true",
                        help="do not use BLEU as metric")
    parser.add_argument("-nM", "--no_METEOR", action="store_true",
                        help="do not use METEOR as metric")
    parser.add_argument("-nR", "--no_ROUGE", action="store_true",
                        help="do not use ROUGE-L as metric")
    parser.add_argument("-nC", "--no_CIDEr", action="store_true",
                        help="do not use CIDEr as metric")

    return parser.parse_args()


def _lc(inputs):
    output = {}

    for k, v in inputs.items():
        output[k] = [s.lower() for s in v]

    return output


class Evaluate(object):

    def __init__(self, bleu=True, meteor=True,
                 rouge=True, cider=True, n=4, lowercase=False):
        self.lc = lowercase
        self.scorers = []

        if bleu:
            if n < 0:
                raise ValueError("n: %d must be a positive integer." % n)

            self.scorers.append(
                (Bleu(n), ["BLEU-%d" % i for i in range(1, n + 1)]))

        if meteor:
            self.scorers.append((Meteor(), "METEOR"))

        if rouge:
            self.scorers.append((Rouge(), "ROUGE-L"))

        if cider:
            self.scorers.append((Cider(), "CIDEr"))

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode("utf-8")

        if isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.items()))

        if isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))

        return data

    def score(self, refs, hypos):
        final_scores = {}

        for scorer, metric in self.scorers:
            score, _ = scorer.compute_score(refs, hypos)

            if isinstance(metric, list):
                for m, s in zip(metric, score):
                    final_scores[m] = s
            else:
                final_scores[metric] = score

        return final_scores

    def evaluate(self, get_scores=True, live=False, **kwargs):
        if live:
            in_refs = kwargs.pop("refs", {})
            in_hypos = kwargs.pop("hypos", {})

            refs = {}
            hypos = {}
            ids = 0

            for k, v in in_hypos.items():
                hypos[ids] = [v]
                refs[ids] = in_refs[k]
                ids += 1
        else:
            refs_files = kwargs.pop("refs", "")
            hypos_file = kwargs.pop("hypos", "")
            refs = {}

            for refs_file in refs_files:
                with open(refs_file) as fd:
                    for ids, line in enumerate(fd):
                        if ids in refs:
                            refs[ids].extend(line.strip().split("\t"))
                        else:
                            refs[ids] = line.strip().split("\t")

            with open(hypos_file) as fd:
                hypos = fd.readlines()
                hypos = {ids: [line.strip()] for ids, line in enumerate(hypos)}

        # whether lowercase?
        if self.lc:
            final_scores = self.score(_lc(refs), _lc(hypos))
        else:
            final_scores = self.score(refs, hypos)

        # output results
        for _, metric in self.scorers:
            if isinstance(metric, list):
                for m in metric:
                    print("%s: %f" % (m, final_scores[m]))
            else:
                print("%s: %f" % (metric, final_scores[metric]))

        if get_scores:
            return final_scores

if __name__ == "__main__":
    args = parse_args()

    if args.no_BLEU and args.no_METEOR and args.no_ROUGE and args.no_CIDEr:
        print("Noting to do, please enable at least one metric!")
        exit(0)

    bleu = not args.no_BLEU
    meteor = not args.no_METEOR
    rouge = not args.no_ROUGE
    cider = not args.no_CIDEr

    obj = Evaluate(bleu=bleu, meteor=meteor,
                   rouge=rouge, cider=cider,
                   n=args.ngram, lowercase=args.lowercase)
    res = obj.evaluate(hypos=args.hypos, refs=args.refs)
