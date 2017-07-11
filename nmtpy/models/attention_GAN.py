# -*- coding: utf-8 -*-
import importlib

from collections import OrderedDict

import theano
import theano.tensor as tensor

import numpy as np
from ..defaults import INT, FLOAT

# Ours
from ..layers import dropout, tanh, get_new_layer
from ..nmtutils import norm_weight
from ..iterators.text import TextIterator
from ..iterators.bitext import BiTextIterator
from .attention import Model as Attention

 
#######################################
## For debugging function input outputs
def inspect_inputs(i, node, fn):
    print('>> Inputs: ', i, node, [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print('>> Outputs: ', i, node, [input[0] for input in fn.outputs])
#######################################

# Same model as attention
class Model(Attention):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)
        
        self.log_probs         = None
        
        # Khoa:
        self.reward = tensor.matrix(name="Reward",dtype=FLOAT)
        # Khoa.

    def load_valid_data(self, from_translate=False):
        self.valid_ref_files = self.data['valid_trg']
        if isinstance(self.valid_ref_files, str):
            self.valid_ref_files = list([self.valid_ref_files])

        if from_translate:
            self.valid_iterator = TextIterator(
                                    mask=False,
                                    batch_size=1,
                                    file=self.data['valid_src'], dict=self.src_dict,
                                    n_words=self.n_words_src)
        else:
            # Take the first validation item for NLL computation
            self.valid_iterator = BiTextIterator(
                                    batch_size=self.batch_size,
                                    srcfile=self.data['valid_src'], srcdict=self.src_dict,
                                    trgfile=self.valid_ref_files[0], trgdict=self.trg_dict,
                                    n_words_src=self.n_words_src, n_words_trg=self.n_words_trg)

        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = BiTextIterator(
                                batch_size=self.batch_size,
                                shuffle_mode=self.smode,
                                logger=self.logger,
                                srcfile=self.data['train_src'], srcdict=self.src_dict,
                                trgfile=self.data['train_trg'], trgdict=self.trg_dict,
                                n_words_src=self.n_words_src,
                                n_words_trg=self.n_words_trg)

        # Prepare batches
        self.train_iterator.read()
        if 'valid_src' in self.data:
            self.load_valid_data()
        
    def build_optimizer(self, cost, reward, regcost, clip_c, dont_update=None, debug=False):
        """Build optimizer by optionally disabling learning for some weights."""
        tparams = OrderedDict(self.tparams)
        
        # Khoa:
        log_probs_ouput = cost[1]
        cost = cost[0]
        # Khoa.
        
        # Filter out weights that we do not want to update during backprop
        if dont_update is not None:
            for key in list(tparams.keys()):
                if key in dont_update:
                    del tparams[key]
        
        
        # Our final cost
        # Khoa:
        final_cost = (reward*log_probs_ouput).sum(0).mean()
#        if reward is not None:
#            final_cost = (reward*log_probs_ouput).sum(0).mean()
#        else:
#            final_cost = cost.mean()
#         Khoa.
        
        # If we have a regularization cost, add it
        if regcost is not None:
            final_cost += regcost


#        # Normalize cost w.r.t sentence lengths to correctly compute perplexity
#        # Only active when y_mask is available
#        if 'y_mask' in self.inputs:
#            norm_cost = (cost / self.inputs['y_mask'].sum(0)).mean()
#            if regcost is not None:
#                norm_cost += regcost
#        else:
#            norm_cost = final_cost

        norm_cost = final_cost
        
        # Get gradients of cost with respect to variables
        # This uses final_cost which is not normalized w.r.t sentence lengths
        grads = tensor.grad(final_cost, wrt=list(tparams.values()))


        # Clip gradients if requested
        if clip_c > 0:
            grads = self.get_clipped_grads(grads, clip_c)

        # Load optimizer
        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]

        # Create theano shared variable for learning rate
        # self.lrate comes from **kwargs / nmt-train params
        self.learning_rate = theano.shared(np.float64(self.lrate).astype(FLOAT), name='lrate')

        # Get updates
        updates = opt(tparams, grads, self.inputs.values(), final_cost, lr0=self.learning_rate)

        # Compile forward/backward function
        if debug:
            self.train_batch = theano.function(list(self.inputs.values())+[reward], norm_cost, updates=updates,
                                               mode=theano.compile.MonitorMode(
                                                   pre_func=inspect_inputs,
                                                   post_func=inspect_outputs), on_unused_input='ignore',allow_input_downcast=True)
        else:
            self.train_batch = theano.function(list(self.inputs.values())+[reward], norm_cost, updates=updates, 
                                               on_unused_input='ignore',allow_input_downcast=True)
            
    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        ############################
        # encoder: bidirectional RNN
        ############################
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)

        # How many additional encoder layers to stack?
        for i in range(1, self.n_enc_layers):
            params = get_new_layer(self.enc_type)[0](params, prefix='deep_encoder_%d' % i,
                                                     nin=self.ctx_dim, dim=self.ctx_dim,
                                                     scale=self.weight_init, layernorm=self.lnorm)

        ############################
        # How do we initialize CGRU?
        ############################
        if self.init_cgru == 'text':
            # init_state computation from mean textual context
            params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)

        #########
        # decoder
        #########
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=self.embedding_dim, dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init, layernorm=False)

        ########
        # fusion
        ########
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'  , nin=self.rnn_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev' , nin=self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'  , nin=self.ctx_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        if self.tied_trg_emb is False:
            params = get_new_layer('ff')[0](params, prefix='ff_logit'  , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)

        self.inputs = OrderedDict()
        self.inputs['x'] = x
        self.inputs['x_mask'] = x_mask
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps = x.shape[0]
        n_timesteps_trg = y.shape[0]
        n_samples = x.shape[1]

        # word embedding for forward rnn (source)
        emb = dropout(self.tparams['Wemb_enc'][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask, layernorm=self.lnorm)

        # word embedding for backward rnn (source)
        embr = dropout(self.tparams['Wemb_enc'][xr.flatten()],
                       self.trng, self.emb_dropout, self.use_dropout)
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask, layernorm=self.lnorm)

        # context will be the concatenation of forward and backward rnns
        ctx = [tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  mask=x_mask, layernorm=self.lnorm)

        # Apply dropout
        ctx = dropout(ctx[0], self.trng, self.ctx_dropout, self.use_dropout)

        if self.init_cgru == 'text':
            # mean of the context (across time) will be used to initialize decoder rnn
            ctx_mean   = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
            init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')
        else:
            # Assume zero-initialized decoder
            init_state = tensor.alloc(0., n_samples, self.rnn_dim)

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = self.tparams['Wemb_dec'][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # decoder - pass through the decoder conditional gru with attention
        proj = get_new_layer('gru_cond')[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state, layernorm=False)
        # hidden states of the decoder gru
        next_state = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # weights (alignment matrix)
        self.alphas = proj[2]

        # compute word probabilities
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')

        logit = dropout(tanh(logit_gru + logit_prev + logit_ctx), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = tensor.dot(logit, self.tparams['Wemb_dec'].T)

        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([n_timesteps_trg, n_samples])
        # Khoa:
        log_probs_ouput = cost
        # Khoa.
        
        cost = (cost * y_mask).sum(0)

        self.log_probs_ouput = theano.function(list(self.inputs.values()), log_probs_ouput)
        
        self.f_log_probs = theano.function(list(self.inputs.values()), cost)

        # For alpha regularization
        # Khoa:
        return cost, log_probs_ouput
    
    
    # Khoa: Return the batch format of sentences and rewards for training
    def get_batch(self,data_values, translated_sentences, discriminator_rewards, professor_rewards ):

        professor_batch_rewards = []
        discriminator_batch_rewards = []
        batch = []
        
        discriminator_rewards = np.array(discriminator_rewards)
        translated_sentences = np.array(translated_sentences)
        
        # Khoa: Similar to mask_data(seqs) of iterator.py, put all translated sentences into a same-size batch
        lengths = [len(s) for s in translated_sentences]
        n_samples = len(translated_sentences)

        maxlen = np.max(lengths) + 1

        # Shape is (t_steps, samples)
        y = np.zeros((n_samples,maxlen,1)).astype(INT)
        y_mask = np.zeros_like(y).astype(FLOAT)
        discriminator_batch_rewards = np.zeros_like(y).astype(FLOAT)
        professor_batch_rewards = np.zeros_like(y).astype(FLOAT)

        for idx, s_y in enumerate(translated_sentences):
            y[idx, :lengths[idx]] = np.array(s_y)
            y_mask[idx, :lengths[idx] + 1] = 1.

        lengths = [len(r) for r in discriminator_rewards]
        for idx, r in enumerate(discriminator_rewards):
            discriminator_batch_rewards[idx, :lengths[idx]] = np.array(r).reshape(lengths[idx],1)
            professor_batch_rewards[idx, :lengths[idx]] = professor_rewards
        
        # Khoa.
        
        # Khoa: Change batch, rewards matrix into good form for function train_batch()
        y_ = y.swapaxes(0,1)
        y_shape = y_.shape
        y = y_.reshape(y_shape[0],y_shape[1])
        
        
        y_mask_ = y_mask.swapaxes(0,1)
        y_mask_shape = y_mask_.shape
        y_mask = y_mask_.reshape(y_mask_shape[0],y_mask_shape[1])
        
        discriminator_batch_rewards_ = discriminator_batch_rewards.swapaxes(0,1)
        discriminator_batch_rewards_shape = discriminator_batch_rewards_.shape
        discriminator_batch_rewards = discriminator_batch_rewards_.reshape(discriminator_batch_rewards_shape[0],discriminator_batch_rewards_shape[1])

        
        professor_batch_rewards_ = professor_batch_rewards.swapaxes(0,1)
        professor_batch_rewards_shape = professor_batch_rewards_.shape
        professor_batch_rewards = professor_batch_rewards_.reshape(professor_batch_rewards_shape[0],professor_batch_rewards_shape[1])
        # Khoa.
        
        batch.append(data_values[0])
        batch.append(data_values[1])
        batch.append(y)
        batch.append(y_mask)
        batch = np.array(batch)
        
        return batch, discriminator_batch_rewards, professor_batch_rewards

    
    # Khoa: Reward for a sentence by using Monte Carlo search 
    def get_reward_MC(self, discriminator, input_sentence, translated_sentence, rollout_num = 20, maxlen = 50, beam_size=12, base_value=0.1):
        final_reward = []
        for token_index in range(len(translated_sentence)):
            if token_index == len(translated_sentence)-1:
                batch = discriminator.get_batch(input_sentence,translated_sentence)
                discriminator_reward = discriminator.get_discriminator_reward(batch[0],batch[1])
                final_reward.append(discriminator_reward[0] - base_value)
            else:
                reward = 0
                max_sentence_len = maxlen - token_index - 1
                for rollout_time  in range(rollout_num):
                    sentence = self.monte_carlo_search(input_sentence,translated_sentence[token_index],[self.f_init],[self.f_next],beam_size,max_sentence_len)
                    sentence_ = np.array(sentence)
                    sentence_shape = sentence_.shape
                    sentence_ = sentence_.reshape(sentence_shape[0],1)
                    
                    final_sentence = np.array(sentence_, dtype=INT)
                    final_sentence = np.concatenate((translated_sentence[0:token_index+1], final_sentence), axis=0)
                   
                    batch = discriminator.get_batch(input_sentence,final_sentence)
                    discriminator_reward = discriminator.get_discriminator_reward(batch[0],batch[1])
                    
                    reward += (discriminator_reward[0] - base_value)
                final_reward.append(reward/rollout_num)
        return np.array(final_reward,dtype=FLOAT)
    
    # Khoa: Reward for a sentence: Discriminator directly assign reward for each parts of token
    def get_reward_not_MC(self, discriminator, input_sentence, translated_sentence, base_value=0.1):
        final_reward = []
        for token_index in range(len(translated_sentence)):
            partially_generated_token = translated_sentence[0:token_index+1]
            batch = discriminator.get_batch(input_sentence,partially_generated_token)
            discriminator_reward = discriminator.get_discriminator_reward(batch[0],batch[1])
            final_reward.append(discriminator_reward[0] - base_value)
        return np.array(final_reward,dtype=FLOAT)
    
    # Khoa: Reward for a sentence: Get reward from Language Model
    def get_reward_LM(self, language_model, translated_sentence, base_value=0.1):
        batch = language_model.get_batch(translated_sentence)
        probs = language_model.pred_probs(batch[0],batch[1])
        return probs
    
    # Khoa: Similar as Beam search but from any token, not just from the first token of a sentence.
    def monte_carlo_search(self, inputs, token, f_inits, f_nexts, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
        
        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []
        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # Number of models
        n_models        = len(f_inits)

        # Ensembling-aware lists
        next_states     = [None] * n_models
        text_ctxs       = [None] * n_models
        aux_ctxs        = [[]] * n_models
        tiled_ctxs      = [None] * n_models
        next_log_ps     = [None] * n_models
        alphas          = [None] * n_models

        for i, f_init in enumerate(f_inits):
            # Get next_state and initial contexts and save them
            # text_ctx: the set of textual annotations
            # aux_ctx: the set of auxiliary (ex: image) annotations
            
            # Khoa:
#            result = list(f_init(*inputs))
            result = list(f_init(inputs))
            #Khoa.
            
            next_states[i], text_ctxs[i], aux_ctxs[i] = result[0], result[1], result[2:]
            tiled_ctxs[i] = np.tile(text_ctxs[i], [1, 1])

        # Beginning-of-sentence indicator is -1
        # Khoa: Start sampling by using the input token
        next_w = token
        #Khoa.
        
        # FIXME: This will break if [0] is not the src sentence, e.g. im2txt modelss
        maxlen = max(maxlen, inputs[0].shape[0] * 3)

        # Initial beam size
        live_beam = beam_size

        for t in range(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)

            # We do this for each model
            for m, f_next in enumerate(f_nexts):
                next_log_ps[m], next_states[m], alphas[m] = f_next(*([next_w, next_states[m], tiled_ctxs[m]] + aux_ctxs[m]))

                if suppress_unks:
                    next_log_ps[m][:, 1] = -np.inf

            # Compute sum of log_p's for the current hypotheses
            cand_scores = hyp_scores[:, None] - sum(next_log_ps)

            # Mean alphas for the mean model (n_models > 1)
            mean_alphas = sum(alphas) / n_models

            # Flatten by modifying .shape (faster)
            cand_scores.shape = cand_scores.size

            # Take the best live_beam hypotheses
            # argpartition makes a partial sort which is faster than argsort
            # (Idea taken from https://github.com/rsennrich/nematus)
            ranks_flat = cand_scores.argpartition(live_beam-1)[:live_beam]

            # Get the costs
            costs = cand_scores[ranks_flat]

            # New states, scores and samples
            live_beam           = 0
            new_hyp_scores      = []
            new_hyp_samples     = []
            new_hyp_alignments  = []

            # This will be the new next states in the next iteration
            hyp_states          = []

            # Find out to which initial hypothesis idx this was belonging
            # Find out the idx of the appended word
            trans_idxs  = ranks_flat // next_log_ps[0].shape[1]
            word_idxs   = ranks_flat % next_log_ps[0].shape[1]

            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_ali = hyp_alignments[ti] + [mean_alphas[ti]]

                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_samples.append(new_hyp)
                    # Cumulated cost of this hypothesis
                    new_hyp_scores.append(costs[idx])
                    new_hyp_alignments.append(new_ali)
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append([next_state[ti] for next_state in next_states])
                    live_beam += 1

            hyp_scores  = np.array(new_hyp_scores, dtype=FLOAT)
            hyp_samples = new_hyp_samples
            hyp_alignments = new_hyp_alignments

            if live_beam == 0:
                break

            # Take the idxs of each hyp's last word
            next_w      = np.array([w[-1] for w in hyp_samples])
            next_states = [np.array(st, dtype=FLOAT) for st in zip(*hyp_states)]
            tiled_ctxs  = [np.tile(ctx, [live_beam, 1]) for ctx in text_ctxs]

        # dump every remaining hypotheses
        for idx in range(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        if not kwargs.get('get_att_alphas', False):
            # Don't send back alignments for nothing
            final_alignments = None

        # Khoa: Return the last value of final_sample
        return final_sample[-1]
     
    # Khoa: The translated sentences could have different sizes (Not ready for a batch)
    def translate(self, inputs, beam_size, maxlen):
        translated_sentences = []
        
        input_sentences = np.array(inputs[0])
        input_sentences = input_sentences.swapaxes(1,0)
        input_sentences_shape = input_sentences.shape
        input_sentences = input_sentences.reshape((input_sentences_shape[0],input_sentences_shape[1],1))
        
        for sentence in input_sentences:
            translated_sentence = self.beam_search_(sentence,[self.f_init],[self.f_next], beam_size, maxlen)
            
            translated_sentence_ = np.array(translated_sentence[0])
            translated_sentence_shape = translated_sentence_.shape
            translated_sentence = translated_sentence_.reshape((translated_sentence_shape[0],1))
            translated_sentences.append(translated_sentence)
        
        return np.array(input_sentences), np.array(translated_sentences)
    
    # Khoa: This beam search is modified and used for function translate() because the errors of f_init.
    def beam_search_(self, inputs, f_inits, f_nexts, beam_size=12, maxlen=100, suppress_unks=False, **kwargs):
        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []
        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # Number of models
        n_models        = len(f_inits)

        # Ensembling-aware lists
        next_states     = [None] * n_models
        text_ctxs       = [None] * n_models
        aux_ctxs        = [[]] * n_models
        tiled_ctxs      = [None] * n_models
        next_log_ps     = [None] * n_models
        alphas          = [None] * n_models

        for i, f_init in enumerate(f_inits):
            # Get next_state and initial contexts and save them
            # text_ctx: the set of textual annotations
            # aux_ctx: the set of auxiliary (ex: image) annotations
            
            #Khoa:
#            result = list(f_init(*inputs))
            result = list(f_init(inputs))
            #Khoa.
            
            next_states[i], text_ctxs[i], aux_ctxs[i] = result[0], result[1], result[2:]
            tiled_ctxs[i] = np.tile(text_ctxs[i], [1, 1])

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,), dtype=INT)

        # FIXME: This will break if [0] is not the src sentence, e.g. im2txt modelss
        maxlen = max(maxlen, inputs[0].shape[0] * 3)

        # Initial beam size
        live_beam = beam_size

        for t in range(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)

            # We do this for each model
            for m, f_next in enumerate(f_nexts):
                next_log_ps[m], next_states[m], alphas[m] = f_next(*([next_w, next_states[m], tiled_ctxs[m]] + aux_ctxs[m]))

                if suppress_unks:
                    next_log_ps[m][:, 1] = -np.inf

            # Compute sum of log_p's for the current hypotheses
            cand_scores = hyp_scores[:, None] - sum(next_log_ps)

            # Mean alphas for the mean model (n_models > 1)
            mean_alphas = sum(alphas) / n_models

            # Flatten by modifying .shape (faster)
            cand_scores.shape = cand_scores.size

            # Take the best live_beam hypotheses
            # argpartition makes a partial sort which is faster than argsort
            # (Idea taken from https://github.com/rsennrich/nematus)
            ranks_flat = cand_scores.argpartition(live_beam-1)[:live_beam]

            # Get the costs
            costs = cand_scores[ranks_flat]

            # New states, scores and samples
            live_beam           = 0
            new_hyp_scores      = []
            new_hyp_samples     = []
            new_hyp_alignments  = []

            # This will be the new next states in the next iteration
            hyp_states          = []

            # Find out to which initial hypothesis idx this was belonging
            # Find out the idx of the appended word
            trans_idxs  = ranks_flat // next_log_ps[0].shape[1]
            word_idxs   = ranks_flat % next_log_ps[0].shape[1]

            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_ali = hyp_alignments[ti] + [mean_alphas[ti]]

                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_samples.append(new_hyp)
                    # Cumulated cost of this hypothesis
                    new_hyp_scores.append(costs[idx])
                    new_hyp_alignments.append(new_ali)
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append([next_state[ti] for next_state in next_states])
                    live_beam += 1

            hyp_scores  = np.array(new_hyp_scores, dtype=FLOAT)
            hyp_samples = new_hyp_samples
            hyp_alignments = new_hyp_alignments

            if live_beam == 0:
                break

            # Take the idxs of each hyp's last word
            next_w      = np.array([w[-1] for w in hyp_samples])
            next_states = [np.array(st, dtype=FLOAT) for st in zip(*hyp_states)]
            tiled_ctxs  = [np.tile(ctx, [live_beam, 1]) for ctx in text_ctxs]

        # dump every remaining hypotheses
        for idx in range(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        if not kwargs.get('get_att_alphas', False):
            # Don't send back alignments for nothing
            final_alignments = None

        return final_sample
            
    # Khoa: The original Beam seach of nmtpy, this function is for nmt-translate (Validation step)
    def beam_search(self, inputs, f_inits, f_nexts, beam_size=12, maxlen=100, suppress_unks=False, **kwargs):
        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []
        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # Number of models
        n_models        = len(f_inits)

        # Ensembling-aware lists
        next_states     = [None] * n_models
        text_ctxs       = [None] * n_models
        aux_ctxs        = [[]] * n_models
        tiled_ctxs      = [None] * n_models
        next_log_ps     = [None] * n_models
        alphas          = [None] * n_models

        for i, f_init in enumerate(f_inits):
            # Get next_state and initial contexts and save them
            # text_ctx: the set of textual annotations
            # aux_ctx: the set of auxiliary (ex: image) annotations
            

            result = list(f_init(*inputs))
            
            next_states[i], text_ctxs[i], aux_ctxs[i] = result[0], result[1], result[2:]
            tiled_ctxs[i] = np.tile(text_ctxs[i], [1, 1])

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,), dtype=INT)

        # FIXME: This will break if [0] is not the src sentence, e.g. im2txt modelss
        maxlen = max(maxlen, inputs[0].shape[0] * 3)

        # Initial beam size
        live_beam = beam_size

        for t in range(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)

            # We do this for each model
            for m, f_next in enumerate(f_nexts):
                next_log_ps[m], next_states[m], alphas[m] = f_next(*([next_w, next_states[m], tiled_ctxs[m]] + aux_ctxs[m]))

                if suppress_unks:
                    next_log_ps[m][:, 1] = -np.inf

            # Compute sum of log_p's for the current hypotheses
            cand_scores = hyp_scores[:, None] - sum(next_log_ps)

            # Mean alphas for the mean model (n_models > 1)
            mean_alphas = sum(alphas) / n_models

            # Flatten by modifying .shape (faster)
            cand_scores.shape = cand_scores.size

            # Take the best live_beam hypotheses
            # argpartition makes a partial sort which is faster than argsort
            # (Idea taken from https://github.com/rsennrich/nematus)
            ranks_flat = cand_scores.argpartition(live_beam-1)[:live_beam]

            # Get the costs
            costs = cand_scores[ranks_flat]

            # New states, scores and samples
            live_beam           = 0
            new_hyp_scores      = []
            new_hyp_samples     = []
            new_hyp_alignments  = []

            # This will be the new next states in the next iteration
            hyp_states          = []

            # Find out to which initial hypothesis idx this was belonging
            # Find out the idx of the appended word
            trans_idxs  = ranks_flat // next_log_ps[0].shape[1]
            word_idxs   = ranks_flat % next_log_ps[0].shape[1]

            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_ali = hyp_alignments[ti] + [mean_alphas[ti]]

                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_samples.append(new_hyp)
                    # Cumulated cost of this hypothesis
                    new_hyp_scores.append(costs[idx])
                    new_hyp_alignments.append(new_ali)
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append([next_state[ti] for next_state in next_states])
                    live_beam += 1

            hyp_scores  = np.array(new_hyp_scores, dtype=FLOAT)
            hyp_samples = new_hyp_samples
            hyp_alignments = new_hyp_alignments

            if live_beam == 0:
                break

            # Take the idxs of each hyp's last word
            next_w      = np.array([w[-1] for w in hyp_samples])
            next_states = [np.array(st, dtype=FLOAT) for st in zip(*hyp_states)]
            tiled_ctxs  = [np.tile(ctx, [live_beam, 1]) for ctx in text_ctxs]

        # dump every remaining hypotheses
        for idx in range(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        if not kwargs.get('get_att_alphas', False):
            # Don't send back alignments for nothing
            final_alignments = None

        return final_sample, final_score, final_alignments
