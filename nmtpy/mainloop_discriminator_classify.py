# -*- coding: utf-8 -*-
from collections import OrderedDict
from shutil import copy

from nmtpy.metrics import is_last_best, find_best, comparators
from nmtpy.sysutils import force_symlink

import numpy as np
import time
import os

class MainLoop(object):
    def __init__(self, model, logger, train_args, model_args):
        # NOTE: model_args not used, if necessary they should be accessible
        # from self.model.*

        # Khoa: Max accuracy of Discriminator
        self.max_acc       = train_args.max_acc
        self.model          = model                         # The model instance that is trained
        self.__log          = logger                        # logger instance

        # Counters
        self.uctr           = 0                             # update ctr
        self.ectr           = 0                             # epoch ctr
        self.vctr           = 0                             # validation ctr
        self.early_bad      = 0                             # early-stop counter

        self.f_snapshot     = train_args.snapshot_freq      # TODO: Modify this for periodic saving
        self.save_best_n    = train_args.save_best_n        # Keep N best validation models on disk
        self.max_updates    = train_args.max_iteration      # Training stops if uctr hits 'max_updates'
        self.max_epochs     = train_args.max_epochs         # Training stops if ectr hits 'max_epochs'

        # Validation related parameters
        self.patience       = train_args.patience           # Stop training if no improvement after this validations
        self.valid_start    = train_args.valid_start        # Start validation at epoch 'valid_start'
        self.beam_size      = train_args.valid_beam         # Beam size for validation decodings
        self.njobs          = train_args.valid_njobs        # # of CPU processes for validation decodings
        self.f_valid        = train_args.valid_freq         # Validation frequency in terms of updates
        self.epoch_valid    = (self.f_valid == 0)           # 0: end of epochs
        self.valid_save_hyp = train_args.valid_save_hyp     # save validation hypotheses under 'valid_hyps' folder
        self.f_verbose      = 10                            # Print frequency

        # TODO: Remove sampling stuff, not useful
        self.f_sample       = train_args.sample_freq
        self.do_sampling    = self.f_sample > 0
        self.n_samples      = 5                             # Number of samples to produce

        self.epoch_losses   = []

        # Multiple comma separated metrics are supported
        # Each key is a metric name, values are metrics so far.
        self.valid_metrics = OrderedDict()

        # We may have no validation data.
        if self.f_valid >= 0:
            # NOTE: This is relevant only for fusion models + WMTIterator
            self.valid_mode = 'single'
            if 'valid_mode' in self.model.__dict__:
                self.valid_mode = self.model.valid_mode

            # Setup validation hypotheses folder name
            if self.valid_save_hyp:
                base_folder = self.model.save_path + '.valid_hyps'
                self.valid_save_prefix = os.path.join(base_folder, os.path.basename(self.model.save_path))

            # Khoa:
#            # Requested metrics, replace px with loss
#            metrics = train_args.valid_metric.replace('px', 'loss').split(',')
#            # first one is for early-stopping
#            self.early_metric = metrics[0]
#            for metric in metrics:
#                self.valid_metrics[metric] = []
            self.early_metric = 'loss'
            # Ensure that loss exists
            self.valid_metrics['loss'] = []

#            # Prepare the string to pass to beam_search
#            self.beam_metrics = ",".join([m for m in self.valid_metrics if m != 'loss'])

            # Best N checkpoint saver
            self.best_models = []

        # FIXME: Disable TB support for now
        self.__tb = None
        ####################
        ## Tensorboard setup
        ####################
#        try:
            #from tensorboard import SummaryWriter
        #except ImportError as ie:
            #self.__tb = None
        #else:
            ## FIXME: This should be a global folder with subfolder
            ## for each training so that we can follow many systems on TB
            #self.__print('Will log training progress to TensorBoard')
            #self.__tb = SummaryWriter(os.path.dirname(self.model.save_path))

    def __send_stats(self, step, **kwargs):
        """Send statistics to TensorBoard."""
        if self.__tb:
            for name, value in kwargs.items():
                if isinstance(value, tuple):
                    # Metric tuple, pass float value
                    value = value[-1]
                self.__tb.add_scalar(name, value, global_step=step)

    def __print(self, msg, footer=False):
        """Pretty prints message with optional separator."""
        self.__log.info(msg)
        if footer:
            self.__log.info('-' * len(msg))



    def run(self):
        """Run training loop."""
        self.model.set_dropout(True)
        #self.model.save(self.model.save_path + '.npz')
        cur_loss = self.model.val_loss()
        self.__print("Loss: %f" % cur_loss)
