# -*- coding: utf-8 -*-
import pickle
import numpy as np

from ..nmtutils     import sent_to_idx
from .iterator      import Iterator
from .homogeneous   import HomogeneousData

# This is an iterator specifically to be used by the .pkl
# corpora files created for WMT16 Shared Task on Multimodal Machine Translation
# Each element of the list that is pickled is in the following format:
# [src_split_idx, trg_split_idx, imgid, imgname, src_words, trg_words]

class WMTIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None, **kwargs):
        super(WMTIterator, self).__init__(batch_size, seed, mask, shuffle_mode, logger)

        assert 'pklfile' in kwargs, "Missing argument pklfile"
        assert 'srcdict' in kwargs, "Missing argument srcdict"

        self._print('Shuffle mode: %s' % shuffle_mode)

        # Short-list sizes
        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        # How do we refer to symbolic data variables?
        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')

        # How do we use the multimodal data? (Numbers in parens are for Task 2)
        # 'all'     : All combinations (~725K parallel)
        # 'single'  : Take only the first pair e.g., train0.en->train0.de (~29K parallel)
        # 'pairs'   : Take only one-to-one pairs e.g., train_i.en->train_i.de (~145K parallel)
        self.mode = kwargs.get('mode', 'pairs')

        # pkl file which contains a list of samples
        self.pklfile = kwargs['pklfile']
        # Resnet-50 image features file
        self.imgfile = kwargs.get('imgfile', None)
        self.img_avail = self.imgfile is not None

        self.trg_avail = False

        # Source word dictionary and short-list limit
        # This may not be available if the task is image -> description (Not implemented)
        self.srcdict = kwargs['srcdict']
        # This may not be available during validation
        self.trgdict = kwargs.get('trgdict', None)

        # Don't use mask when batch_size == 1 which means we're doing
        # translation with nmt-translate
        if self.batch_size == 1:
            self.mask = False

        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        # We have images in the middle
        if self.imgfile:
            self._keys.append("%s_img" % self.src_name)

        # Target may not be available during validation
        if self.trgdict:
            self._keys.append(self.trg_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        # Load image features file if any
        if self.img_avail:
            self._print('Loading image file...')
            self.img_feats = np.load(self.imgfile)
            self._print('Done.')

        # Load the corpora
        with open(self.pklfile, 'rb') as f:
            self._print('Loading pkl file...')
            self._seqs = pickle.load(f)
            self._print('Done.')

        # Check for what is available
        ss = self._seqs[0]
        # If no split idxs are found, its Task 1, set mode to 'all'
        if ss[0] is None and ss[1] is None:
            self.mode = 'all'

        if ss[5] is not None and self.trgdict:
            self.trg_avail = True

        if self.mode == 'single':
            # Just take the first src-trg pair. Useful for validation
            if ss[1] is not None:
                self._seqs = [s for s in self._seqs if (s[0] == s[1] == 0)]
            else:
                self._seqs = [s for s in self._seqs if (s[0] == 0)]

        elif ss[1] is not None and self.mode == 'pairs':
            # Take the pairs with split idx's equal
            self._seqs = [s for s in self._seqs if s[0] == s[1]]

        # We now have a list of samples
        self.n_samples = len(self._seqs)

        # Depending on mode, we can have multiple sentences per image so
        # let's store the number of actual images as well.
        # n_unique_samples <= n_samples
        self.n_unique_images = len(set([s[3] for s in self._seqs]))

        # Some statistics
        total_src_words = []
        total_trg_words = []

        # Let's map the sentences once to idx's
        for sample in self._seqs:
            sample[4] = sent_to_idx(self.srcdict, sample[4], self.n_words_src)
            total_src_words.extend(sample[4])
            if self.trg_avail:
                sample[5] = sent_to_idx(self.trgdict, sample[5], self.n_words_trg)
                total_trg_words.extend(sample[5])

        self.unk_src = total_src_words.count(1)
        self.unk_trg = total_trg_words.count(1)
        self.total_src_words = len(total_src_words)
        self.total_trg_words = len(total_trg_words)

        #########################
        # Prepare iteration stuff
        #########################
        # Set batch processor function
        if self.batch_size == 1:
            self._process_batch = (lambda idxs: self.process_single(idxs[0]))
        else:
            self._process_batch = (lambda idxs: self.mask_seqs(idxs))

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=5)
        else:
            # For once keep it ordered
            self._idxs = np.arange(self.n_samples).tolist()
            self._iter = []
            for i in range(0, self.n_samples, self.batch_size):
                self._iter.append(self._idxs[i:i + self.batch_size])
            self._iter = iter(self._iter)

    def process_single(self, idx):
        data, _ = Iterator.mask_data([self._seqs[idx][4]])
        data = [data]
        if self.img_avail:
            # Do this 196 x 1024
            data += [self.img_feats[self._seqs[idx][2]][:, None, :]]
        if self.trg_avail:
            trg, _ = Iterator.mask_data([self._seqs[idx][5]])
            data.append(trg)
        return data

    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
        data = list(Iterator.mask_data([self._seqs[i][4] for i in idxs]))
        # Source image features
        if self.img_avail:
            img_idxs = [self._seqs[i][2] for i in idxs]

            # Do this 196 x bsize x 1024
            x_img = self.img_feats[img_idxs].transpose(1, 0, 2)
            data += [x_img]

        if self.trg_avail:
            data += list(Iterator.mask_data([self._seqs[i][5] for i in idxs]))

        return data

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            # Fill in the _idxs list for sample order
            if self.shuffle_mode == 'simple':
                # Simple shuffle
                self._idxs = np.random.permutation(self.n_samples).tolist()
            elif self.shuffle_mode is None:
                # Ordered
                self._idxs = np.arange(self.n_samples).tolist()

            self._iter = []
            for i in range(0, self.n_samples, self.batch_size):
                self._iter.append(self._idxs[i:i + self.batch_size])
            self._iter = iter(self._iter)
