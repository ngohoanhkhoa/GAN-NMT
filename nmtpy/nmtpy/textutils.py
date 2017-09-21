# -*- coding: utf-8 -*-
"""Text processing related functions"""

def reduce_to_best(hyps, scores, n_unique_samples, avoid_unk=True):
    """Pick the best of each hypotheses group based on their scores."""

    if avoid_unk:
        # Penalize hyps having <unk> inside
        pairs = [(p[0], p[1] + (100 if "<unk>" in p[0][0] else 0)) for p in zip(hyps, scores)]

    # Group each sample's hypotheses
    groups = [pairs[i::n_unique_samples] for i in range(n_unique_samples)]

    # Now each element of "groups" contain let's say 5 hypotheses and their scores
    # Sort them and get the first (smallest score)
    return [sorted(g, key=lambda x: x[1])[0][0] for g in groups]
