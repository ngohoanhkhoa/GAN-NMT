# -*- coding: utf-8 -*-

from nmtpy.mainloop import MainLoop
import os

class MainLoop(MainLoop):
    def __init__(self, model, logger, train_args, model_args):
        super(MainLoop, self).__init__(model, logger, train_args, model_args)
        
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
            self.early_metric = 'loss'
            
            # Ensure that loss exists
            self.valid_metrics['loss'] = []
            self.best_models = []
