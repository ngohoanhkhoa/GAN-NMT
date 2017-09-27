# -*- coding: utf-8 -*-

from nmtpy.mainloop import MainLoop

import numpy as np
import time

# Khoa: From the class MainLoop of mainloop.py
class MainLoop(MainLoop):
    def __init__(self, model, discriminator, logger, train_args, model_args):
        # Call parent's init first
        super(MainLoop, self).__init__(model, logger, train_args, model_args)
        
        # NOTE: model_args not used, if necessary they should be accessible
        # from self.model.*
        
        # Khoa:
        self.discriminator  = discriminator
        self.best_discriminator = discriminator
        
        self.monte_carlo_search = train_args.monte_carlo_search
        self.maxlen = train_args.maxlen
        self.rollnum = train_args.rollnum
        
        # Number of loop for Generator and Discriminator
        self.generator_loop_num = train_args.generator_loop_num
        self.discriminator_reward_loop_num = train_args.discriminator_reward_loop_num
        self.discriminator_loop_num = train_args.discriminator_loop_num
        self.professor_forcing_loop_num = train_args.professor_forcing_loop_num
        
        # Maximum - Minimum accuracy of Discriminator
        self.max_acc = train_args.max_acc
        self.min_acc = train_args.min_acc
        
        self.professor_forcing_reward = train_args.professor_forcing_reward
        
        # Khoa.
    
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
            
    # Khoa: Apply GANs
    def __train_epoch(self):
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
            
            #Train the generator
            for it in range(self.generator_loop_num):
                # Khoa: Generate samples
                # Khoa: def translate(self, inputs, beam_size = 1 , maxlen), Be careful with beam_size != 1 => Error
                input_sentences, translated_sentences, translated_states = self.model.translate_beam_search(
                                                                             inputs = list(data.values()),
                                                                             maxlen = self.maxlen)


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
                                                                            professor_rewards= self.professor_forcing_reward)
                
                

                rewards = discriminator_rewards
                
                
                # -------------------------------------------------------------
                for it in range(self.discriminator_reward_loop_num):
                    # Khoa: Update Generator with Reward from Discriminator
                    # (Using machine-translated sentence)
                    loss_generator = self.model.train_batch_discriminator_reward(*batch_generator, rewards)
                    self.__print('Loss Generator D: %f' % loss_generator)
                    generator_batch_losses.append(loss_generator)
                    self.__send_stats(self.uctr, train_loss=loss_generator)
                
                
                # Khoa: Update Generator with Professor Forcing (Using human-translated sentence)
                for it in range(self.professor_forcing_loop_num):
                    loss_generator = self.model.train_batch_professor_forcing(*list(data.values()), professor_rewards)
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
                    self.__print("Discriminator: Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss_discriminator))
                    discriminator_batch_losses.append(loss_discriminator)

            # Verbose
            if self.uctr % self.f_verbose == 0:
                self.__print("Generator    : Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss_generator))
                #self.__print("Discriminator: Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss_discriminator))
                
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

