# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Provides:
cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences)
into a form usable by score_cooked().
"""

import copy
import math
from collections import defaultdict


def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return len(words), counts


def cook_refs(refs, eff=None, n=4):  # lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    """

    reflen = []
    maxcounts = {}

    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen))/len(reflen)

    # lhuang: N.B.: leave reflen computaiton to the very end!!

    # lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return reflen, maxcounts


def cook_test(test, ref_len_counts, eff=None, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""
    reflen, refmaxcounts = ref_len_counts
    testlen, counts = precook(test, n, True)
    result = {}

    # Calculate effective reference sentence length.
    if eff == "closest":
        result["reflen"] = min((abs(l - testlen), l) for l in reflen)[1]
    else:
        result["reflen"] = reflen

    result["testlen"] = testlen
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
    result["correct"] = [0 for _ in range(n)]

    for ngram, count in counts.items():
        result["correct"][len(ngram) - 1] += min(
            refmaxcounts.get(ngram,0), count)

    return result


class BleuScorer(object):
    """BLEU scorer."""

    __slots__ = "n", "crefs", "ctest", "_score", \
        "_ratio", "_testlen", "_reflen", "special_reflen"

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    # special_reflen is used in oracle
    # (proportional effective ref len for a node).
    def copy(self):
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None

        return new

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ 
        to avoid creating new instances."""

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)
            else:
                self.ctest.append(None)

        self._score = None

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        return self.fscore(option=option), self.ratio(option=option)

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if isinstance(new_test, str):
            new_test = [new_test]

        assert len(new_test) == len(self.crefs), new_test

        self.ctest = []

        for t, r in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, r))

        self._score = None

        return self

    def rescore(self, new_test):
        return self.retest(new_test).compute_score()

    def size(self):
        num_refs = len(self.crefs)
        num_test = len(self.ctest)

        if num_test != num_refs:
            raise ValueError("test(%d)/refs(%d)"
                             " mismatch!" % (num_refs, num_test))

        return num_refs

    def __iadd__(self, other):
        if isinstance(other, tuple):
            # avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            if not self.compatible(other):
                raise ValueError("incompatible BLEUs")

            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):

        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            raise ValueError("Unknown reflen option %s" % option)

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {
            "testlen": 0,
            "reflen": 0,
            "guess": [0 for _ in range(n)],
            "correct": [0 for _ in range(n)]
        }

        # for each sentence
        for comps in self.ctest:
            testlen = comps["testlen"]
            self._testlen += testlen

            if self.special_reflen is None:
                reflen = self._single_reflen(comps["reflen"], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ["guess", "correct"]:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.0

            for k in range(n):
                bleu *= (float(comps["correct"][k]) + tiny) \
                        / (float(comps["guess"][k]) + small)
                bleu_list[k].append(bleu ** (1.0 / (k + 1)))

            ratio = (testlen + tiny) / (reflen + small)

            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1.0 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps["reflen"] = self._reflen
        totalcomps["testlen"] = self._testlen

        bleus = []
        bleu = 1.0

        for k in range(n):
            bleu *= float(totalcomps["correct"][k] + tiny) \
                    / (totalcomps["guess"][k] + small)
            bleus.append(bleu ** (1.0 / (k + 1)))

        ratio = (self._testlen + tiny) / (self._reflen + small)

        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1.0 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio: %f" % ratio)

        # Normalize to percentage
        bleus = [100 * b for b in bleus]
        self._score = bleus

        return self._score, bleu_list
