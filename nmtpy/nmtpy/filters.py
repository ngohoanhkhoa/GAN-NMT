# -*- coding: utf-8 -*-
import re

class CompoundFilter(object):
    """Filters out fillers from compound splitted sentences."""
    def __init__(self):
        pass

    def __filter(self, s):
        return s.replace(" @@ ", "").replace(" @@", "").replace(" @", "").replace("@ ", "")

    def __call__(self, inp):
        if isinstance(inp, str):
            return self.__filter(inp)
        else:
            return [self.__filter(e) for e in inp]

class BPEFilter(object):
    """Filters out fillers from BPE applied sentences."""
    def __init__(self):
        pass

    def __filter(self, s):
        # The first replace misses lines ending with @@
        # like 'foo@@ bar Hotel@@'
        return s.replace("@@ ", "").replace("@@", "")

    def __call__(self, inp):
        if isinstance(inp, str):
            return self.__filter(inp)
        else:
            return [self.__filter(e) for e in inp]

class DesegmentFilter(object):
    """Converts Turkish segmentations of <tag:morpheme> to normal form."""
    def __init__(self):
        pass

    def __filter(self, s):
        return re.sub(' *<.*?:(.*?)>', '\\1', s)

    def __call__(self, inp):
        if isinstance(inp, str):
            return self.__filter(inp)
        else:
            return [self.__filter(e) for e in inp]

def get_filter(name):
    filters = {
                "bpe"          : BPEFilter(),
                "compound"     : CompoundFilter(),
                "desegment"    : DesegmentFilter(),
              }
    return filters.get(name, None)
