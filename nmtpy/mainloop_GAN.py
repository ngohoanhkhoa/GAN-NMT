# -*- coding: utf-8 -*-
from collections import OrderedDict

from nmtpy.metrics import is_last_best, find_best, comparators
from nmtpy.sysutils import force_symlink
from nmtpy.nmtutils import idx_to_sent

import numpy as np
import time
import os

class MainLoop(object):
    def __init__(self, model, discriminator, language_model, logger, train_args, model_args):
        # NOTE: model_args not used, if necessary they should be accessible
        # from self.model.*
        
        # Khoa:
        if discriminator is not None:
            self.discriminator  = discriminator
            self.best_discriminator = discriminator
            
        self.language_model =  language_model
        
        # Alpha value for modifying the reward between Discriminator and Language Model
        self.alpha = train_args.alpha_init
        self.alpha_rate = train_args.alpha_rate
        
        self.monte_carlo_search = train_args.monte_carlo_search
        self.maxlen = train_args.maxlen
        self.rollnum = train_args.rollnum
        
        # Number of loop for Generator and Discriminator
        self.generator_loop_num = train_args.generator_loop_num
        self.discriminator_loop_num = train_args.discriminator_loop_num
        
        # Maximum - Minimum accuracy of Discriminator
        
        self.max_acc = train_args.max_acc
        self.min_acc = train_args.min_acc
        # Khoa.
        
        # Khoa: The model of generator - Our main NMT model
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

            # Requested metrics, replace px with loss
            metrics = train_args.valid_metric.replace('px', 'loss').split(',')

            # first one is for early-stopping
            self.early_metric = metrics[0]

            for metric in metrics:
                self.valid_metrics[metric] = []

            # Ensure that loss exists
            self.valid_metrics['loss'] = []

            # Prepare the string to pass to beam_search
            self.beam_metrics = ",".join([m for m in self.valid_metrics if m != 'loss'])

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

    def __save_best_model(self):
        """Saves best N models to disk."""
        if self.save_best_n > 0:
            # Get the score of the system that will be saved
            cur_score = self.valid_metrics[self.early_metric][-1]

            # Custom filename with metric score
            cur_fname = "%s-val%3.3d-%s_%.3f.npz" % (self.model.save_path, self.vctr, self.early_metric, cur_score)

            # Stack is empty, save the model whatsoever
            if len(self.best_models) < self.save_best_n:
                self.best_models.append((cur_score, cur_fname))

            # Stack is full, replace the worst model
            else:
                os.unlink(self.best_models[self.next_prune_idx][1])
                self.best_models[self.next_prune_idx] = (cur_score, cur_fname)

            self.__print('Saving model with best validation %s' % self.early_metric.upper())
            self.model.save(cur_fname)

            # Create a .BEST symlink
            force_symlink(cur_fname, ('%s.BEST.npz' % self.model.save_path), relative=True)

            # In the next best, we'll remove the following idx from the list/disk
            # Metric specific comparator stuff
            where = comparators[self.early_metric][-1]
            self.next_prune_idx = sorted(range(len(self.best_models)),
                                         key=self.best_models.__getitem__)[where]

    def __update_lrate(self):
        """Update learning rate by annealing it."""
        pass
    

    def __do_sampling(self, data):
        """Generates samples and prints them."""
        if self.do_sampling and self.uctr % self.f_sample == 0:
            samples = self.model.generate_samples(data, self.n_samples)
            if samples is not None:
                for src, truth, sample in samples:
                    if src:
                        self.__print("Source: %s" % src)
                    self.__print.info("Sample: %s" % sample)
                    self.__print.info(" Truth: %s" % truth)

    def __do_validation(self):
        """Do early-stopping validation."""
        if self.ectr >= self.valid_start:
            self.vctr += 1

            # Compute validation loss
            self.model.set_dropout(False)
            cur_loss = self.model.val_loss()
            self.model.set_dropout(True)

            # Add val_loss
            self.valid_metrics['loss'].append(cur_loss)

            # Print validation loss
            self.__print("Validation %2d - LOSS = %.3f (PPL: %.3f)" % (self.vctr, cur_loss, np.exp(cur_loss)))

            #############################
            # Are we doing beam search? #
            #############################
            if self.beam_metrics:
                beam_results = None
                # Save beam search results?
                f_valid_out = None

                if self.valid_save_hyp:
                    f_valid_out = "{0}.{1:03d}".format(self.valid_save_prefix, self.vctr)

                self.__print('Calling beam-search process')
                beam_time = time.time()
                beam_results = self.model.run_beam_search(beam_size=self.beam_size,
                                                          n_jobs=self.njobs,
                                                          metric=self.beam_metrics,
                                                          mode='beamsearch',
                                                          valid_mode=self.valid_mode,
                                                          f_valid_out=f_valid_out)
                beam_time = time.time() - beam_time
                self.__print('Beam-search ended, took %.5f minutes.' % (beam_time / 60.))

                if beam_results:
                    # beam_results: {name: (metric_str, metric_float)}
                    # names are as defined in metrics/*.py like BLEU, METEOR
                    # but we use lowercase names in conf files.
                    self.__send_stats(self.vctr, **beam_results)
                    for name, (metric_str, metric_value) in beam_results.items():
                        self.__print("Validation %2d - %s" % (self.vctr, metric_str))
                        self.valid_metrics[name.lower()].append(metric_value)
                else:
                    self.__print('Skipping this validation since nmt-translate probably failed.')
                    # Return back to training loop since nmt-translate did not run correctly.
                    # This will allow to fix your model's build_sampler while training continues.
                    return

            # Is this the best evaluation based on early-stop metric?
            if is_last_best(self.early_metric, self.valid_metrics[self.early_metric]):
                if self.valid_save_hyp:
                    # Create a link towards best hypothesis file
                    force_symlink(f_valid_out, '%s.BEST' % self.valid_save_prefix, relative=True)

                self.__save_best_model()
                self.early_bad = 0
            else:
                self.early_bad += 1
                self.__print("Early stopping patience: %d validation left" % (self.patience - self.early_bad))

            self.__dump_val_summary()
    
    def __do_validation_check_accuracy_discriminator(self):
        prob_true = []
        for data in self.model.valid_iterator:
            if self.monte_carlo_search:
                # Khoa: def prepare_data_MC(self, data_values, generator, beam_size=1, maxlen=50)
                batch_discriminator = self.discriminator.prepare_data_MC(list(data.values()), self.model)
            else:
                # Khoa: def prepare_data_not_MC(self, data_values, generator, beam_size = 1, maxlen=50)
                batch_discriminator = self.discriminator.prepare_data_not_MC(list(data.values()), self.model)
    
            probs = self.discriminator.get_probs_valid(*batch_discriminator)
            probs = np.array(probs)*np.array(batch_discriminator[2])
            probs = probs.sum(1)
            true_num= sum(1 for prob in probs if prob > 0.5)
            prob_true.append((true_num/len(probs)))
            
        mean_acc = np.array(prob_true).mean()
        
        if mean_acc < self.max_acc and mean_acc > self.min_acc:
            self.best_discriminator = self.discriminator
            return True
        
        if mean_acc < self.min_acc:
            return True
        
        if mean_acc > self.max_acc:
            self.discriminator = self.best_discriminator
            return False
            
            
    def __dump_val_summary(self):
        """Print validation summary."""
        for metric, history in self.valid_metrics.items():
            if len(history) > 0:
                # Find the best validation idx and value so far
                best_idx, best_val = find_best(metric, history)
                if metric == 'loss':
                    msg = "BEST %s = %.3f (PPL: %.3f)" % (metric.upper(), best_val, np.exp(best_val))
                else:
                    msg = "BEST %s = %.3f" % (metric.upper(), best_val)

                self.__print('--> Current %s at validation %d' % (msg, best_idx))

        # Remember who we are
        self.__print('--> This is model: %s' % os.path.basename(self.model.save_path))

    def __dump_epoch_summary(self, losses, epoch_time, up_ctr):
        """Print epoch summary."""
        update_time = epoch_time / float(up_ctr)
        mean_loss = np.array(losses).mean()
        self.epoch_losses.append(mean_loss)

        self.__print("--> Epoch %d finished with mean loss %.5f (PPL: %4.5f)" % (self.ectr, mean_loss, np.exp(mean_loss)))
        self.__print("--> Epoch took %.3f minutes, %.3f sec/update" % ((epoch_time / 60.0), update_time))

    def run(self):
        """Run training loop."""
        self.model.set_dropout(True)
        #self.model.save(self.model.save_path + '.npz')
#        while self.__train_epoch():
        # Khoa:
        while self.__train_GAN(): pass
        # Khoa.
        
        # Final summary
        self.__print('Saving last model found.')
        self.model.save("%s_last.npz" % self.model.save_path)
        
        if self.f_valid >= 0:
            self.__dump_val_summary()
        else:
            # No validation data used, save the final model
            self.__print('Saving final model.')
            self.model.save("%s.npz" % self.model.save_path)
            

    def __train_GAN(self):
        """Train a full epoch."""
        self.ectr += 1

        start = time.time()
        start_uctr = self.uctr
        self.__print('Starting Epoch %d' % self.ectr, True)

        generator_batch_losses = []
        discriminator_batch_losses = []
        
        #Iterate over batches
        for data in self.model.train_iterator:
            self.uctr += 1
            
            if self.alpha <= 1:
                self.alpha += self.alpha_rate

            #Train the generator
            for it in range(self.generator_loop_num):
                # Khoa: Generate samples
                # Khoa: def translate(self, inputs, beam_size = 1 , maxlen), Be careful with beam_size != 1 => Error
                input_sentences, translated_sentences, translated_states = self.model.translate_beam_search(
                                                                             inputs = list(data.values()),
                                                                             maxlen = self.maxlen)

#                print('true y', list(data.values())[2])
#                print('pred y', translated_sentences)

                # Khoa: Get reward for each sentence in batch. 
                # -------------------------------------------------------------
                # Khoa: Reward from Discriminator
                # There are two ways of Discriminator: 
                    # Monte Carlo search (MC) or Getting directly from Discriminator (not_MC)
                discriminator_rewards_ = []
                for (input_sentence, translated_sentence, translated_state)  in zip(input_sentences, translated_sentences, translated_states):
                    if self.monte_carlo_search:
                        # Khoa: def get_reward_MC(self, discriminator, 
                        # input_sentence, translated_sentence, translated_states, 
                        # rollout_num = 20, maxlen = 50, base_value=0.0)

                        reward = self.model.get_reward_MC(discriminator       = self.discriminator, 
                                                          input_sentence      = input_sentence, 
                                                          translated_sentence = translated_sentence,
                                                          translated_states   = translated_state,
                                                          rollout_num         = self.rollnum, 
                                                          maxlen              = self.maxlen, 
                                                          base_value          = 0.0)
                        
                        
                    else:
                        # Khoa: def get_reward_not_MC(self, discriminator, 
                        # input_sentence, translated_sentence, base_value=0.0):
                        
                        reward = self.model.get_reward_not_MC(discriminator       = self.discriminator, 
                                                              input_sentence      = input_sentence, 
                                                              translated_sentence = translated_sentence,
                                                              base_value          = 0.0)
                    
                    discriminator_rewards_.append(reward)
                        
                # Khoa: def get_batch(self,data_values, translated_sentences, 
                # discriminator_rewards, professor_rewards,language_model_rewards)
                batch_generator, discriminator_rewards, professor_rewards = self.model.get_batch(
                                                                            list(data.values()), 
                                                                            translated_sentences, 
                                                                            discriminator_rewards=discriminator_rewards_,
                                                                            professor_rewards= 1)
                
                # -------------------------------------------------------------
                # Khoa: Reward of Language Model
                # Reward for each token in sentence
                language_model_rewards = []
                if self.language_model is not None:
                    language_model_rewards_ = []
                    for translated_sentence in translated_sentences:
                        # Khoa: Reward when a full sentence is put into LM.
                        # def get_reward_LM(self, language_model, translated_sentence, base_value=0.0)
                        #reward = self.model.get_reward_LM(language_model = self.language_model, 
                        #                                translated_sentence = translated_sentence, 
                        #                                base_value=0.0)
                        
                        # Khoa: Reward when partially generated sentences are put into LM.
                        reward = self.model.get_reward_partial_LM(language_model = self.language_model, 
                                                        translated_sentence = translated_sentence, 
                                                        base_value=0.0)
                        
                        language_model_rewards_.append(reward)
                        

                    language_model_rewards = self.model.get_batch_reward_for_lm(translated_sentences, 
                                                                                language_model_rewards_)
                
                # -------------------------------------------------------------
                if self.language_model is not None:
                    rewards = self.alpha*discriminator_rewards + (1-self.alpha)*language_model_rewards
                else:
                    rewards = discriminator_rewards
                
                # -------------------------------------------------------------
                # Khoa: Update Generator with Reward from Discriminator or/and Language Model 
                # (Using machine-translated sentence)
                
#                a = self.model.log_probs_output(*batch_generator)
#                b = a*rewards
#                print('Reward: ', rewards)
#                print('Not reward: ', a.sum(0).mean())
#                print('Dis: ', b.sum(0).mean())
#                
#                a = self.model.log_probs_output(*list(data.values()))
#                b = a*professor_rewards
#                print('Pro: ', b.sum(0).mean())
#                
#                print('final_cost: ', self.model.final_cost(*list(data.values()), professor_rewards) )
#                print('get_cost: ', self.model.get_cost(*list(data.values()), professor_rewards) )
#                print('get_norm_cost: ', self.model.get_norm_cost(*list(data.values()), professor_rewards) )
                
                loss_generator = self.model.train_batch(*batch_generator, rewards)
                self.__print('Loss Generator D: %f' % loss_generator)
                generator_batch_losses.append(loss_generator)
                self.__send_stats(self.uctr, train_loss=loss_generator)
                
                # Khoa: Update Generator with Professor Forcing (Using human-translated sentence)
                loss_generator = self.model.train_batch(*list(data.values()), professor_rewards)
                
                # Khoa: Get loss
                self.__print('Loss Generator P: %f' % loss_generator)
                generator_batch_losses.append(loss_generator)
                self.__send_stats(self.uctr, train_loss=loss_generator)

                
            # Train de discriminator
            if self.__do_validation_check_accuracy_discriminator():
                for it in range(self.discriminator_loop_num):
                    if self.monte_carlo_search:
                        # Khoa: def prepare_data_MC(self, data_values, generator, maxlen=50)
                        batch_discriminator = self.discriminator.prepare_data_MC(list(data.values()), self.model)
                    else:
                        # Khoa: def prepare_data_not_MC(self, data_values, generator, maxlen=50)
                        batch_discriminator = self.discriminator.prepare_data_not_MC(list(data.values()), self.model)
                    
                    # Update Discriminator
                    loss_discriminator = self.discriminator.train_batch(*batch_discriminator)
                    
                    # Khoa: Get loss
                    self.__print('Loss Discriminator: %10.6f' % loss_discriminator)
                    discriminator_batch_losses.append(loss_discriminator)

            # Verbose
            if self.uctr % self.f_verbose == 0:
                self.__print("Generator    : Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss_generator))
                self.__print("Discriminator: Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss_discriminator))
                
            # Should we stop
            if self.uctr == self.max_updates:
                self.__print("Max iteration %d reached." % self.uctr)
                return False

            # Update learning rate if requested
            self.__update_lrate()

            # Do sampling
            self.__do_sampling(data)

            # Do validation
            if not self.epoch_valid and self.f_valid > 0 and self.uctr % self.f_valid == 0:
                self.__do_validation()

            # Check stopping conditions
            if self.early_bad == self.patience:
                self.__print("Early stopped.")
                return False

        # An epoch is finished
        epoch_time = time.time() - start

        # Print epoch summary
        up_ctr = self.uctr - start_uctr
        self.__print("---------------------------------------------------------")
        self.__print("Epoch summary of Generator    :")
        self.__dump_epoch_summary(generator_batch_losses, epoch_time, up_ctr)
        self.__print("Epoch summary of Discriminator:")
        self.__dump_epoch_summary(discriminator_batch_losses, epoch_time, up_ctr)
        self.__print("---------------------------------------------------------")
        
        # Do validation
        if self.epoch_valid:
            self.__do_validation()

        # Check whether maximum epoch is reached
        if self.ectr == self.max_epochs:
            self.__print("Max epochs %d reached." % self.max_epochs)
            return False

        return True

