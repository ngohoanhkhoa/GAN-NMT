# -*- coding: utf-8 -*-

from nmtpy.metrics import is_last_best
from nmtpy.sysutils import force_symlink
from nmtpy.mainloop import MainLoop

import time
import os


class MainLoop(MainLoop):
    def __init__(self, model, logger, train_args, model_args):
        # Call parent's init first
        super(MainLoop, self).__init__(model, logger, train_args, model_args)

        # Khoa: Max accuracy of Discriminator
        self.max_acc       = train_args.max_acc
        # Khoa.

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
            # Khoa.

#            # Prepare the string to pass to beam_search
#            self.beam_metrics = ",".join([m for m in self.valid_metrics if m != 'loss'])

            # Best N checkpoint saver
            self.best_models = []

        # FIXME: Disable TB support for now
        self.__tb = None



    def __train_epoch(self):
        """Train a full epoch."""
        self.ectr += 1

        start = time.time()
        start_uctr = self.uctr
        self.__print('Starting Epoch %d' % self.ectr, True)

        batch_losses = []

        # Iterate over batches
        for data in self.model.train_iterator:
            self.uctr += 1

            # Forward/backward and get loss
            batch_discriminator = self.model.get_batch(x = list(data.values())[0],
                                                       y = list(data.values())[2],
                                                       label = list(data.values())[4])

            loss = self.model.train_batch(*batch_discriminator)
            batch_losses.append(loss)
            self.__send_stats(self.uctr, train_loss=loss)

            # Verbose
            if self.uctr % self.f_verbose == 0:
                self.__print("Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss))

            # Should we stop
            if self.uctr == self.max_updates:
                self.__print("Max iteration %d reached." % self.uctr)
                return False

            # Update learning rate if requested
            self.__update_lrate()

            # Do validation
            # Khoa: Remember to check early stop condition in self.__do_validation()
            if not self.epoch_valid and self.f_valid > 0 and self.uctr % self.f_valid == 0:
                if not self.__do_validation():
                    return False

            # Check stopping conditions
            if self.early_bad == self.patience:
                self.__print("Early stopped.")
                return False

        # An epoch is finished
        epoch_time = time.time() - start

        # Print epoch summary
        up_ctr = self.uctr - start_uctr
        self.__dump_epoch_summary(batch_losses, epoch_time, up_ctr)

        # Do validation
        if self.epoch_valid:
            if not self.__do_validation():
                return False

        # Check whether maximum epoch is reached
        if self.ectr == self.max_epochs:
            self.__print("Max epochs %d reached." % self.max_epochs)
            return False

        return True

    def __do_sampling(self, data):
        return 0

    def __do_validation(self):
        """Do early-stopping validation."""
        if self.ectr >= self.valid_start:
            self.vctr += 1

            # Compute validation loss
            # self.model.set_dropout(False)
            cur_loss = self.model.val_loss()
            # self.model.set_dropout(True)
            
            # Add val_loss
            self.valid_metrics['loss'].append(cur_loss)
            
            # Print validation loss
            self.__print("Validation %2d - ACC = %.3f (LOSS: %.3f)" % (self.vctr, 1.0 - cur_loss, cur_loss) )
            
            f_valid_out = None
            if self.valid_save_hyp:
                    f_valid_out = "{0}.{1:03d}".format(self.valid_save_prefix, self.vctr)
   
            if is_last_best('loss', self.valid_metrics['loss']):
                if self.valid_save_hyp:
                    # Create a link towards best hypothesis file
                    force_symlink(f_valid_out, '%s.BEST' % self.valid_save_prefix, relative=True)
                    
                self.__save_best_model()
                self.early_bad = 0
            else:
                self.early_bad += 1
                self.__print("Early stopping patience: %d validation left" % (self.patience - self.early_bad))
                
            self.__dump_val_summary()
            # Khoa: Set the initial accuracy for Discriminator in GAN
            if cur_loss < (1 - self.max_acc):
                self.__print("Reach maximum accuracy %.3f : Current Accuracy = %.3f " % (self.max_acc,1-cur_loss ))
                return False
            else:
                return True
            
        return True
            
