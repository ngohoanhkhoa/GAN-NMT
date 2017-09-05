# -*- coding: utf-8 -*-
import random
import pickle
from collections import OrderedDict

import numpy as np

from .iterator import Iterator

class FlickrIterator(object):
    """Iterator for Karpathy's DeepSent dataset."""
    def __init__(self, pkl_file, pkl_split, batch_size, trg_dict, n_words_trg=0, src_name='x_img', trg_name='y'):
        # For minibatch shuffling
        random.seed(1234)

        self.trg_dict = trg_dict
        self.n_words_trg = n_words_trg

        self.pkl_file = pkl_file
        self.split = pkl_split

        self.batch_size = batch_size
        self.shuffle = False

        self.src_name = src_name
        self.trg_name = trg_name

        self.n_samples = 0
        self.__seqs = []
        self.__minibatches = []
        self.__return_keys = [self.src_name, self.trg_name, "%s_mask" % self.trg_name]
        self.__iter = None

        self.read()

    def __repr__(self):
        return "%s (split: %s)" % (self.pkl_file, self.split)

    def set_batch_size(self, bs):
        self.batch_size = bs
        self.prepare_batches()

    def rewind(self):
        if self.shuffle:
            random.shuffle(self.__minibatches)

        self.__iter = iter(self.__minibatches)

    def __iter__(self):
        return self

    def read(self):
        def to_idx(tokens):
            idxs = []
            for w in tokens:
                # Get token, 1 if not available
                widx = self.trg_dict.get(w, 1)
                if self.n_words_trg > 0:
                    widx = widx if widx < self.n_words_trg else 1
                idxs.append(widx)
            return idxs

        ##############
        with open(self.pkl_file, 'rb') as f:
            d = pickle.load(f)

        self.feats = d['feats']

        # hackish way to understand the dimensions
        if self.feats.shape[0] < self.feats.shape[1]:
            # feat_dim is 0, n_samples 1
            self.feats = self.feats.T

        self.img_dim = self.feats.shape[1]

        # NOTE: Add ability to read multiple splits which will
        # help during final training on both train and valid
        # Make this by checking the type of pkl_split against str or list
        sents = d['sents'][self.split]

        # feats has size (feat_dim, n_samples)
        # sents: list of dict (train: 29000, test: 1000)
        self.__seqs = []
        for x in sents:
            # 5 captions
            for p in x['sentences']:
                self.__seqs.append([p['imgid'], to_idx(p['tokens'])])
                # Only take the first sentences for test/valid
                if self.split == 'test':
                    break

        # Save sentence count
        self.n_samples = len(self.__seqs)

    def prepare_batches(self, shuffle=False):
        if shuffle:
            self.shuffle = True
            random.shuffle(self.__seqs)

        self.__minibatches = []
        batches = [list(range(i, min(i+self.batch_size, self.n_samples))) \
                        for i in range(0, self.n_samples, self.batch_size)]

        for idxs in batches:
            x = np.vstack(self.feats[self.__seqs[i][0]] for i in idxs)
            y, y_mask = Iterator.mask_data([self.__seqs[i][1] for i in idxs])
            self.__minibatches.append((x, y, y_mask))

        self.__iter = iter(self.__minibatches)

    def __next__(self):
        try:
            data = next(self.__iter)
        except StopIteration as si:
            self.rewind()
            raise
        except AttributeError as ae:
            raise Exception("You need to call prepare_batches() first.")
        else:
            return OrderedDict([(k,data[i]) for i,k in enumerate(self.__return_keys)])
