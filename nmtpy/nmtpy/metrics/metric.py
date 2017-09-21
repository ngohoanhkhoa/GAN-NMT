# -*- coding: utf-8 -*-
from functools import total_ordering

@total_ordering
class Metric(object):
    def __init__(self, score=None):
        self.score_str = "0.0"
        self.score = 0.
        self.name = ""

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return "%s = %s" % (self.name, self.score_str)
