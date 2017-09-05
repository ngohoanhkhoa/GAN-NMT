# -*- coding: utf-8 -*-
import numpy as np
import copy

# Iterator that randomly fetches samples with same target
# length to be efficient in terms of RNN underlyings.
# Code from https://github.com/kelvinxu/arctic-captions
class HomogeneousData(object):
    def __init__(self, data, batch_size, trg_pos):
        self.batch_size = batch_size
        self.data = data
        self.trg_pos = trg_pos

        self.prepare()
        self.reset()

    def prepare(self):
        # find all target sequence lengths
        self.lengths = [len(cc[self.trg_pos]) for cc in self.data]

        # Compute unique lengths
        self.len_unique = np.unique(self.lengths)

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()

        # For each length, find the sample idxs and their counts
        for ll in self.len_unique:
            self.len_indices[ll] = np.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)

        # Randomize length order
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            # Randomize sample order for a specific length
            self.len_indices[ll] = np.random.permutation(self.len_indices[ll])
            # Set initial position for this length to 0
            self.len_indices_pos[ll] = 0

        self.len_idx = -1

    def __next__(self):
        fin_unique_len = 0
        while True:
            # What is the length idx for this batch?
            self.len_idx = (self.len_idx + 1) % len(self.len_unique)
            # Current candidate length
            self.cur_len = self.len_unique[self.len_idx]
            # Do we have samples left for this length?
            if self.len_curr_counts[self.cur_len] > 0:
                break

            # All samples for a length exhausted, increment counter
            fin_unique_len += 1

            # Is this the end for this epoch?
            if fin_unique_len >= len(self.len_unique):
                break

        # All data consumed
        if fin_unique_len >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # batch_size or what is left for this length
        curr_batch_size = np.minimum(self.batch_size, self.len_curr_counts[self.cur_len])
        # Get current position for the batch
        curr_pos = self.len_indices_pos[self.cur_len]

        # get the indices for the current batch
        curr_indices = self.len_indices[self.cur_len][curr_pos:curr_pos+curr_batch_size]

        # Increment next position
        self.len_indices_pos[self.cur_len] += curr_batch_size
        # Decrement used sample count
        self.len_curr_counts[self.cur_len] -= curr_batch_size

        # Return batch indices from here
        return curr_indices

    def __iter__(self):
        return self
