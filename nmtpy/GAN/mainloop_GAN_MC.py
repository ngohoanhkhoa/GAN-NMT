# -*- coding: utf-8 -*-

from nmtpy.nmtutils import idx_to_sent
from mainloop_GAN import MainLoop

import numpy as np
import time



import pickle as pickle

# Khoa: From the class MainLoop of mainloop_GAN.py
class MainLoop(MainLoop):
    def __init__(self, model, discriminator, language_model, logger, train_args, model_args):
        super(MainLoop, self).__init__(model, logger, train_args, model_args)
        
        # Khoa:
        self.mc_research_directory = train_args.mc_research
        # Khoa.

    def __train_epoch(self):
        """Train a full epoch."""
        self.ectr += 1

        start = time.time()
        start_uctr = self.uctr
        self.__print('Starting Epoch %d' % self.ectr, True)

        generator_batch_losses = []
        discriminator_batch_losses = []
        
                
        #Iterate over batches
        # Khoa:
        batch_num = 0
        # Khoa.
        for data in self.model.train_iterator:
            self.uctr += 1
            # Khoa:
            batch_num += 1
            file_string = []
            # Khoa.
            if self.alpha <= 1:
                self.alpha += self.alpha_rate
            
            #Train the generator
            for it in range(self.generator_loop_num):
                # Khoa: Generate samples
                # Khoa: def translate(self, inputs, beam_size = 1 , maxlen)
                input_sentences, translated_sentences, translated_states = self.model.translate_beam_search(
                                                                             list(data.values()),
                                                                             maxlen=self.maxlen)
                
                 # Khoa: Get reward for each sentence in batch. 
                # -------------------------------------------------------------
                # Khoa: Reward from Discriminator
                # There are two ways of Discriminator: 
                    # Monte Carlo search (MC) or Getting directly from Discriminator (not_MC)
                discriminator_rewards_ = []
                for (input_sentence, translated_sentence, translated_state)  in zip(input_sentences, translated_sentences, translated_states):
                    if self.monte_carlo_search: 
                        # def get_reward_MC_research(self, discriminator, generator, 
                        # input_sentence, translated_sentence, translated_states, 
                        # rollout_num = 20, maxlen = 50, base_value=0.1):
                        reward, reward_research = self.get_reward_MC_research(
                                                    discriminator       = self.discriminator,
                                                    generator           = self.model,
                                                    input_sentence      = input_sentence,
                                                    translated_sentence = translated_sentence, 
                                                    translated_states   = translated_state,
                                                    rollout_num         = self.rollnum, 
                                                    maxlen              = self.maxlen,
                                                    base_value          = 0.5)
                        
                        
                        input_sentence_string = []
                        for token in np.array(input_sentence):
                            token_string = idx_to_sent(self.model.src_idict, token)
                            input_sentence_string.append(token_string)
                            
                        output_sentence_string = []
                        for token in np.array(translated_sentence):
                            token_string = idx_to_sent(self.model.trg_idict, token)
                            output_sentence_string.append(token_string)
                            
                        reward_string = np.array(reward_research)
                        
                        file_string.append([input_sentence_string, output_sentence_string, reward_string])

#                        reward = self.model.get_reward_MC(self.discriminator, 
#                                                   input_sentence, 
#                                                   translated_sentence,
#                                                   translated_state,
#                                                   rollout_num = self.rollnum, 
#                                                   maxlen = self.maxlen, 
#                                                   base_value=0.5)
                    else:
                        # Khoa: def get_reward_not_MC(self, discriminator, 
                        # input_sentence, translated_sentence, base_value=0.1)
                        
                        reward = self.model.get_reward_not_MC(self.discriminator, 
                                                   input_sentence, 
                                                   translated_sentence,
                                                   base_value=0.5)
                    
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
                        # Khoa: def get_reward_LM(self, language_model, translated_sentence, base_value=0.1)
                        reward = self.model.get_reward_LM(language_model = self.language_model, 
                                                        translated_sentence = translated_sentence, 
                                                        base_value=0.1)
                        
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
                loss_generator = self.model.train_batch(*batch_generator, rewards)
                
                # Khoa: Update Generator with Professor Forcing (Using human-translated sentence)
                loss_generator = self.model.train_batch(*list(data.values()), professor_rewards)
                
                # Khoa: Get loss
                self.__print('Loss Generator: %10.6f' % loss_generator)
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
                    self.__print('Loss Discriminaror: %10.6f' % loss_discriminator)
                    discriminator_batch_losses.append(loss_discriminator)
            
            #----------------------------------------------------------------------
            with open(self.mc_research_directory + '/MC_file_' + str(self.ectr) +  '_' + str(batch_num)  + '.txt', 'wb') as f:
                pickle.dump(file_string, f)
            print('file: ', '/MC_file_' + str(self.ectr) +  '_' + str(batch_num)  + '.txt')
            #----------------------------------------------------------------------

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

    # Khoa: Reward for a sentence by using Monte Carlo search 
    def get_reward_MC_research(self, discriminator, generator, input_sentence, translated_sentence, translated_states, rollout_num = 20, maxlen = 50, base_value=0.1):
        final_reward = []
        final_reward_token = []
       
        for token_index in range(len(translated_sentence)):
            if token_index == len(translated_sentence)-1:
                batch = discriminator.get_batch(input_sentence,translated_sentence)
                discriminator_reward = discriminator.get_discriminator_reward(batch[0],batch[1])
                final_reward.append(discriminator_reward[0] - base_value)
                final_reward_token.append(discriminator_reward[0])
            else:
                reward_research = []
                reward = 0
                max_sentence_len = maxlen - token_index - 1
                for rollout_time  in range(rollout_num):
                    sentence = generator.sampling_multinomial(inputs = input_sentence, 
                                                         token = translated_sentence[token_index], 
                                                         state = translated_states[token_index], 
                                                         f_init = generator.f_init,
                                                         f_next = generator.f_next,
                                                         maxlen = max_sentence_len)
                    sentence_ = np.array(sentence)
                    sentence_shape = sentence_.shape
                    sentence_ = sentence_.reshape(sentence_shape[0],1)
                    
                    final_sentence = np.array(sentence_, dtype='int64')
                    final_sentence = np.concatenate((translated_sentence[0:token_index+1], final_sentence), axis=0)
                    
                    batch = discriminator.get_batch(input_sentence,final_sentence)
                    discriminator_reward = discriminator.get_discriminator_reward(batch[0],batch[1])
                    
                    reward_research.append(discriminator_reward[0])
                    reward += (discriminator_reward[0] - base_value)
                    
                final_reward.append(reward/rollout_num)
                final_reward_token.append(reward_research)
        

        return np.array(final_reward,dtype='float32'), final_reward_token