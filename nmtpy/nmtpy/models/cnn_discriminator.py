# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import importlib

import theano
import theano.tensor as tensor
from ..layers import get_new_layer
from ..defaults import INT, FLOAT
from ..nmtutils import load_dictionary, norm_weight, pp
from ..nmtutils import invert_dictionary
# Ours
from .basemodel import BaseModel
from ..layers import dropout

from ..sysutils   import fopen
from ..iterators.iterator    import Iterator
from ..iterators.homogeneous import HomogeneousData

#######################################
## For debugging function input outputs
def inspect_inputs(i, node, fn):
    print('>> Inputs: ', i, node, [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print('>> Outputs: ', i, node, [input[0] for input in fn.outputs])
#######################################


class Model(BaseModel):
    def __init__(self, seed, logger, **kwargs):
        super(Model, self).__init__(**kwargs)
        
        # Shuffle mode (default: No shuffle)
        self.smode = kwargs.get('shuffle_mode', 'simple')
        # Load dictionaries
        if 'src_dict' in kwargs:
            # Already passed through kwargs (nmt-translate)
            self.src_dict = kwargs['src_dict']
            # Invert dict
            src_idict = invert_dictionary(self.src_dict)
        else:
            # Load them from pkl files
            self.src_dict, src_idict = load_dictionary(kwargs['dicts']['src'])

        if 'trg_dict' in kwargs:
            # Already passed through kwargs (nmt-translate)
            self.trg_dict = kwargs['trg_dict']
            # Invert dict
            trg_idict = invert_dictionary(self.trg_dict)
        else:
            # Load them from pkl files
            self.trg_dict, trg_idict = load_dictionary(kwargs['dicts']['trg'])

        # Limit shortlist sizes
        self.n_words_src = min(self.n_words_src, len(self.src_dict)) \
                if self.n_words_src > 0 else len(self.src_dict)
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) \
                if self.n_words_trg > 0 else len(self.trg_dict)

        # Create options. This will saved as .pkl
        self.set_options(self.__dict__)

        self.trg_idict = trg_idict
        self.src_idict = src_idict
        
        # Set the seed of Theano RNG
        self.set_trng(seed)
        # We call this once to setup dropout mechanism correctly
        self.set_dropout(False)
        
        self.logger = logger
        
        self.emb_dropout = kwargs.get('emb_dropout', 0.)
        
        
    def load_valid_data(self, from_translate=False):
        self.valid_iterator = ClassificationIterator(
                                batch_size=self.batch_size,
                                srcfile=self.data['valid_src'], srcdict=self.src_dict,
                                trgfile=self.data['valid_trg'], trgdict=self.trg_dict,
                                labelfile=self.data['valid_label'],
                                n_words_src=self.n_words_src,
                                n_words_trg=self.n_words_trg)

        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = ClassificationIterator(
                                batch_size=self.batch_size,
                                shuffle_mode=self.smode,
                                logger=self.logger,
                                srcfile=self.data['train_src'], srcdict=self.src_dict,
                                trgfile=self.data['train_trg'], trgdict=self.trg_dict,
                                labelfile=self.data['train_label'],
                                n_words_src=self.n_words_src,
                                n_words_trg=self.n_words_trg)

        # Prepare batches
        self.train_iterator.read()
        if 'valid_src' in self.data:
            self.load_valid_data()
            

    def init_params(self):
        params = OrderedDict()
        
        # embedding weights for encoder
        params['Wemb_enc_x'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_enc_y'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)


        # input_shape : (input_channels, input_rows, input_cols)
        # filter_shape: (output_channels, input_channels, filter_rows, filter_cols)
        # Check layers.py and nmtutils.py
        #Output: batch x (SenLen - 3 + 1) X 1
        
        input_shape = (None, None, None)
        
        # Khoa: Convolution for x
        filter_shape = (1,1,3,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x_1')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_x_1', nin=48, nout=10, scale=self.weight_init)
        
        
        filter_shape = (1,1,5,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x_2')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_x_2', nin=46, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,7,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x_3')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_x_3', nin=44, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,9,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x_4')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_x_4', nin=42, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,11,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x_5')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_x_5', nin=40, nout=10, scale=self.weight_init)
        
        # Khoa: Convolution for y
        filter_shape = (1,1,3,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y_1')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_y_1', nin=48, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,5,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y_2')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_y_2', nin=46, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,7,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y_3')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_y_3', nin=44, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,9,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y_4')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_y_4', nin=42, nout=10, scale=self.weight_init)
        
        filter_shape = (1,1,11,self.embedding_dim)
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y_5')
        params = get_new_layer('ff')[0](params, prefix='ff_conv_y_5', nin=40, nout=10, scale=self.weight_init)
        
        params = get_new_layer('ff')[0](params, prefix='ff_logit', nin=50, nout=2, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        y = tensor.matrix('y', dtype=INT)
        label = tensor.matrix('label', dtype=FLOAT)
        
        self.inputs = OrderedDict()
        self.inputs['x'] = x
        self.inputs['y'] = y
        self.inputs['label'] = label

        n_timesteps = x.shape[0]
        n_timesteps_trg = y.shape[0]
        n_samples = x.shape[1]

        # word embedding
        emb_x = dropout(self.tparams['Wemb_enc_x'][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb_x = emb_x.reshape([n_timesteps, n_samples, self.embedding_dim])
        
        emb_y = dropout(self.tparams['Wemb_enc_y'][y.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb_y = emb_y.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        
        # Khoa: Change to 4D tensor
        emb_x = emb_x.swapaxes(0,1)
        emb_x_shape = emb_x.shape
        emb_x = emb_x.reshape((emb_x_shape[0],1,emb_x_shape[1],emb_x_shape[2]),ndim=4)
        
        emb_y = emb_y.swapaxes(0,1)
        emb_y_shape = emb_y.shape
        emb_y = emb_y.reshape((emb_y_shape[0],1,emb_y_shape[1],emb_y_shape[2]),ndim=4)
        
        
        # Khoa: Convolution for x
        conv_x_1 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_1', activ='relu')
        conv_x_2 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_2', activ='relu')
        conv_x_3 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_3', activ='relu')
        conv_x_4 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_4', activ='relu')
        conv_x_5 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_5', activ='relu')
        
        # Khoa: Convolution for y
        conv_y_1 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_1', activ='relu')
        conv_y_2 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_2', activ='relu')
        conv_y_3 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_3', activ='relu')
        conv_y_4 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_4', activ='relu')
        conv_y_5 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_5', activ='relu')
        
        # Khoa: Transform to 3D tensor for x
        conv_x_1_ = conv_x_1.swapaxes(0,1)
        conv_x_1_shape = conv_x_1_.shape
        conv_x_1_ = conv_x_1_.reshape((conv_x_1_shape[0],conv_x_1_shape[1],conv_x_1_shape[2]),ndim=3)
        
        conv_x_2_ = conv_x_2.swapaxes(0,1)
        conv_x_2_shape = conv_x_2_.shape
        conv_x_2_ = conv_x_2_.reshape((conv_x_2_shape[0],conv_x_2_shape[1],conv_x_2_shape[2]),ndim=3)
        
        conv_x_3_ = conv_x_3.swapaxes(0,1)
        conv_x_3_shape = conv_x_3_.shape
        conv_x_3_ = conv_x_3_.reshape((conv_x_3_shape[0],conv_x_3_shape[1],conv_x_3_shape[2]),ndim=3)
        
        conv_x_4_ = conv_x_4.swapaxes(0,1)
        conv_x_4_shape = conv_x_4_.shape
        conv_x_4_ = conv_x_4_.reshape((conv_x_4_shape[0],conv_x_4_shape[1],conv_x_4_shape[2]),ndim=3)
        
        conv_x_5_ = conv_x_5.swapaxes(0,1)
        conv_x_5_shape = conv_x_5_.shape
        conv_x_5_ = conv_x_5_.reshape((conv_x_5_shape[0],conv_x_5_shape[1],conv_x_5_shape[2]),ndim=3)
        
        # Khoa: Transform to 3D tensor for y
        conv_y_1_ = conv_y_1.swapaxes(0,1)
        conv_y_1_shape = conv_y_1_.shape
        conv_y_1_ = conv_y_1_.reshape((conv_y_1_shape[0],conv_y_1_shape[1],conv_y_1_shape[2]),ndim=3)
        
        conv_y_2_ = conv_y_2.swapaxes(0,1)
        conv_y_2_shape = conv_y_2_.shape
        conv_y_2_ = conv_y_2_.reshape((conv_y_2_shape[0],conv_y_2_shape[1],conv_y_2_shape[2]),ndim=3)
        
        conv_y_3_ = conv_y_3.swapaxes(0,1)
        conv_y_3_shape = conv_y_3_.shape
        conv_y_3_ = conv_y_3_.reshape((conv_y_3_shape[0],conv_y_3_shape[1],conv_y_3_shape[2]),ndim=3)
        
        conv_y_4_ = conv_y_4.swapaxes(0,1)
        conv_y_4_shape = conv_y_4_.shape
        conv_y_4_ = conv_y_4_.reshape((conv_y_4_shape[0],conv_y_4_shape[1],conv_y_4_shape[2]),ndim=3)
        
        conv_y_5_ = conv_y_5.swapaxes(0,1)
        conv_y_5_shape = conv_y_5_.shape
        conv_y_5_ = conv_y_5_.reshape((conv_y_5_shape[0],conv_y_5_shape[1],conv_y_5_shape[2]),ndim=3)
        
        # Khoa: Apply ff to conv_x, transform into 10 dim
        ff_conv_x_1 = get_new_layer('ff')[1](self.tparams, conv_x_1_, prefix='ff_conv_x_1', activ='linear')
        ff_conv_x_2 = get_new_layer('ff')[1](self.tparams, conv_x_2_, prefix='ff_conv_x_2', activ='linear')
        ff_conv_x_3 = get_new_layer('ff')[1](self.tparams, conv_x_3_, prefix='ff_conv_x_3', activ='linear')
        ff_conv_x_4 = get_new_layer('ff')[1](self.tparams, conv_x_4_, prefix='ff_conv_x_4', activ='linear')
        ff_conv_x_5 = get_new_layer('ff')[1](self.tparams, conv_x_5_, prefix='ff_conv_x_5', activ='linear')

        # Khoa: Apply ff to conv_y, transform into 10 dim
        ff_conv_y_1 = get_new_layer('ff')[1](self.tparams, conv_y_1_, prefix='ff_conv_y_1', activ='linear')
        ff_conv_y_2 = get_new_layer('ff')[1](self.tparams, conv_y_2_, prefix='ff_conv_y_2', activ='linear')
        ff_conv_y_3 = get_new_layer('ff')[1](self.tparams, conv_y_3_, prefix='ff_conv_y_3', activ='linear')
        ff_conv_y_4 = get_new_layer('ff')[1](self.tparams, conv_y_4_, prefix='ff_conv_y_4', activ='linear')
        ff_conv_y_5 = get_new_layer('ff')[1](self.tparams, conv_y_5_, prefix='ff_conv_y_5', activ='linear')
        
        # Khoa: Concatenate x and y features
        concatenated_conv = self.concatenate([ff_conv_x_1,ff_conv_x_2,ff_conv_x_3,ff_conv_x_4,ff_conv_x_5,
                                   ff_conv_y_1,ff_conv_y_2,ff_conv_y_3,ff_conv_y_4,ff_conv_y_5],axis=2 )
    
    
        concatenated_conv_ = concatenated_conv.swapaxes(0,1)
        concatenated_conv_shape = concatenated_conv_.shape
        concatenated_conv_ = concatenated_conv_.reshape((concatenated_conv_shape[0],
                                                         concatenated_conv_shape[1],
                                                        concatenated_conv_shape[2]),ndim=3)
        
        # Khoa: Max pooling
        conv_pool_size = (1,2)
        conv_after_pooling = tensor.signal.pool.pool_2d(input=concatenated_conv_,ws=conv_pool_size,
                                                        ignore_border=True)
    
        
        # Khoa: Swap axes
        conv_after_pooling_ = conv_after_pooling.swapaxes(0,1)
        conv_after_pooling_shape = conv_after_pooling_.shape
        conv_after_pooling_ = conv_after_pooling_.reshape((conv_after_pooling_shape[0],
                                                           conv_after_pooling_shape[1],
                                                        conv_after_pooling_shape[2]),ndim=3)
        
        
        # Khoa: Put into ff, from 50 to 1 -> sigmoid -> cost binary_crossentropy
        # Khoa: Put into ff, from 50 to 2 -> softmqx -> cost categorical_crossentropy
        # Khoa: Put into ff, from 50 to 2
        logit = get_new_layer('ff')[1](self.tparams, conv_after_pooling_, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        
        
        # Khoa: Apply softmax 
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[1], logit_shp[0]*logit_shp[2]]))
        
        # Khoa: Avoid the error of nan in binary_crossentropy, probs could be 0. or 1. so log(0.), log(1.)
        probs = tensor.clip(probs, 1e-7, 1.0 - 1e-7)
        
        # Khoa: Get binary cross entropy
#        cost = tensor.nnet.binary_crossentropy(probs, label)
        # Khoa: The same value in cols, so just take max
#        cost = cost.max(1)

        # Khoa: Get categorical crossentropy
        cost = tensor.nnet.categorical_crossentropy(probs, label)
        
        self.get_probs_valid = theano.function(list(self.inputs.values()), probs, on_unused_input='ignore')
        
        self.get_cost = theano.function(list(self.inputs.values()), cost, on_unused_input='warn')


        return cost
    
    def build_optimizer(self, cost, regcost, clip_c, dont_update=None, debug=False):
        """Build optimizer by optionally disabling learning for some weights."""
        tparams = OrderedDict(self.tparams)
        
        
        # Filter out weights that we do not want to update during backprop
        if dont_update is not None:
            for key in list(tparams.keys()):
                if key in dont_update:
                    del tparams[key]
        
        
        # Our final cost
        final_cost = cost.mean()
        
        
        # If we have a regularization cost, add it
        if regcost is not None:
            final_cost += regcost


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
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates,
                                               mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs), 
                                            on_unused_input='warn')
        else:
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates, 
                                               on_unused_input='warn')
    
    def build_sampler(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        y = tensor.matrix('y', dtype=INT)

        n_timesteps = x.shape[0]
        n_timesteps_trg = y.shape[0]
        n_samples = x.shape[1]

        # word embedding
        emb_x = dropout(self.tparams['Wemb_enc_x'][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb_x = emb_x.reshape([n_timesteps, n_samples, self.embedding_dim])
        
        emb_y = dropout(self.tparams['Wemb_enc_y'][y.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb_y = emb_y.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        
        # Khoa: Change to 4D tensor
        emb_x = emb_x.swapaxes(0,1)
        emb_x_shape = emb_x.shape
        emb_x = emb_x.reshape((emb_x_shape[0],1,emb_x_shape[1],emb_x_shape[2]),ndim=4)
        
        emb_y = emb_y.swapaxes(0,1)
        emb_y_shape = emb_y.shape
        emb_y = emb_y.reshape((emb_y_shape[0],1,emb_y_shape[1],emb_y_shape[2]),ndim=4)
        
        
        # Khoa: Convolution for x
        conv_x_1 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_1', activ='relu')
        conv_x_2 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_2', activ='relu')
        conv_x_3 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_3', activ='relu')
        conv_x_4 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_4', activ='relu')
        conv_x_5 = self.conv_layer(self.tparams, emb_x, prefix='conv_x_5', activ='relu')
        
        # Khoa: Convolution for y
        conv_y_1 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_1', activ='relu')
        conv_y_2 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_2', activ='relu')
        conv_y_3 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_3', activ='relu')
        conv_y_4 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_4', activ='relu')
        conv_y_5 = self.conv_layer(self.tparams, emb_y, prefix='conv_y_5', activ='relu')
        
        # Khoa: Transform to 3D tensor for x
        conv_x_1_ = conv_x_1.swapaxes(0,1)
        conv_x_1_shape = conv_x_1_.shape
        conv_x_1_ = conv_x_1_.reshape((conv_x_1_shape[0],conv_x_1_shape[1],conv_x_1_shape[2]),ndim=3)
        
        conv_x_2_ = conv_x_2.swapaxes(0,1)
        conv_x_2_shape = conv_x_2_.shape
        conv_x_2_ = conv_x_2_.reshape((conv_x_2_shape[0],conv_x_2_shape[1],conv_x_2_shape[2]),ndim=3)
        
        conv_x_3_ = conv_x_3.swapaxes(0,1)
        conv_x_3_shape = conv_x_3_.shape
        conv_x_3_ = conv_x_3_.reshape((conv_x_3_shape[0],conv_x_3_shape[1],conv_x_3_shape[2]),ndim=3)
        
        conv_x_4_ = conv_x_4.swapaxes(0,1)
        conv_x_4_shape = conv_x_4_.shape
        conv_x_4_ = conv_x_4_.reshape((conv_x_4_shape[0],conv_x_4_shape[1],conv_x_4_shape[2]),ndim=3)
        
        conv_x_5_ = conv_x_5.swapaxes(0,1)
        conv_x_5_shape = conv_x_5_.shape
        conv_x_5_ = conv_x_5_.reshape((conv_x_5_shape[0],conv_x_5_shape[1],conv_x_5_shape[2]),ndim=3)
        
        # Khoa: Transform to 3D tensor for y
        conv_y_1_ = conv_y_1.swapaxes(0,1)
        conv_y_1_shape = conv_y_1_.shape
        conv_y_1_ = conv_y_1_.reshape((conv_y_1_shape[0],conv_y_1_shape[1],conv_y_1_shape[2]),ndim=3)
        
        conv_y_2_ = conv_y_2.swapaxes(0,1)
        conv_y_2_shape = conv_y_2_.shape
        conv_y_2_ = conv_y_2_.reshape((conv_y_2_shape[0],conv_y_2_shape[1],conv_y_2_shape[2]),ndim=3)
        
        conv_y_3_ = conv_y_3.swapaxes(0,1)
        conv_y_3_shape = conv_y_3_.shape
        conv_y_3_ = conv_y_3_.reshape((conv_y_3_shape[0],conv_y_3_shape[1],conv_y_3_shape[2]),ndim=3)
        
        conv_y_4_ = conv_y_4.swapaxes(0,1)
        conv_y_4_shape = conv_y_4_.shape
        conv_y_4_ = conv_y_4_.reshape((conv_y_4_shape[0],conv_y_4_shape[1],conv_y_4_shape[2]),ndim=3)
        
        conv_y_5_ = conv_y_5.swapaxes(0,1)
        conv_y_5_shape = conv_y_5_.shape
        conv_y_5_ = conv_y_5_.reshape((conv_y_5_shape[0],conv_y_5_shape[1],conv_y_5_shape[2]),ndim=3)
        
        # Khoa: Apply ff to conv_x, transform into 10 dim
        ff_conv_x_1 = get_new_layer('ff')[1](self.tparams, conv_x_1_, prefix='ff_conv_x_1', activ='linear')
        ff_conv_x_2 = get_new_layer('ff')[1](self.tparams, conv_x_2_, prefix='ff_conv_x_2', activ='linear')
        ff_conv_x_3 = get_new_layer('ff')[1](self.tparams, conv_x_3_, prefix='ff_conv_x_3', activ='linear')
        ff_conv_x_4 = get_new_layer('ff')[1](self.tparams, conv_x_4_, prefix='ff_conv_x_4', activ='linear')
        ff_conv_x_5 = get_new_layer('ff')[1](self.tparams, conv_x_5_, prefix='ff_conv_x_5', activ='linear')

        # Khoa: Apply ff to conv_y, transform into 10 dim
        ff_conv_y_1 = get_new_layer('ff')[1](self.tparams, conv_y_1_, prefix='ff_conv_y_1', activ='linear')
        ff_conv_y_2 = get_new_layer('ff')[1](self.tparams, conv_y_2_, prefix='ff_conv_y_2', activ='linear')
        ff_conv_y_3 = get_new_layer('ff')[1](self.tparams, conv_y_3_, prefix='ff_conv_y_3', activ='linear')
        ff_conv_y_4 = get_new_layer('ff')[1](self.tparams, conv_y_4_, prefix='ff_conv_y_4', activ='linear')
        ff_conv_y_5 = get_new_layer('ff')[1](self.tparams, conv_y_5_, prefix='ff_conv_y_5', activ='linear')
        
        # Khoa: Concatenate x and y features
        concatenated_conv = self.concatenate([ff_conv_x_1,ff_conv_x_2,ff_conv_x_3,ff_conv_x_4,ff_conv_x_5,
                                   ff_conv_y_1,ff_conv_y_2,ff_conv_y_3,ff_conv_y_4,ff_conv_y_5],axis=2 )
    
    
        concatenated_conv_ = concatenated_conv.swapaxes(0,1)
        concatenated_conv_shape = concatenated_conv_.shape
        concatenated_conv_ = concatenated_conv_.reshape((concatenated_conv_shape[0],
                                                         concatenated_conv_shape[1],
                                                        concatenated_conv_shape[2]),ndim=3)
        
        # Khoa: Max pooling
        conv_pool_size = (1,2)
        conv_after_pooling = tensor.signal.pool.pool_2d(input=concatenated_conv_,ws=conv_pool_size,
                                                        ignore_border=True)
    
        
        # Khoa: Swap axes
        conv_after_pooling_ = conv_after_pooling.swapaxes(0,1)
        conv_after_pooling_shape = conv_after_pooling_.shape
        conv_after_pooling_ = conv_after_pooling_.reshape((conv_after_pooling_shape[0],
                                                           conv_after_pooling_shape[1],
                                                        conv_after_pooling_shape[2]),ndim=3)
        
        # Khoa: Put into ff, from 50 to 2
        logit = get_new_layer('ff')[1](self.tparams, conv_after_pooling_, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        
        # Khoa: Apply softmax 
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[1], logit_shp[0]*logit_shp[2]]))
        
        self.get_probs = theano.function(list([x,y]), probs, on_unused_input='warn')
    
    # Khoa: Get the probability of a sentence being human-translated
    def get_discriminator_reward(self, x,y):
        probs = self.get_probs(x,y)
        probs  = np.array(probs)
        return probs[:,0]
    
    def prepare_data_MC(self, data_values, generator, maxlen=50):
        input_sentences, translated_sentences = generator.translate_multinomial(data_values, maxlen)
        
        translated_sentences_ = []
        for sentence in translated_sentences:
            # Khoa: Modify (Increase and decrease) the size of translated_sentences_ into sentence lenght = 50 for convolution
            # Similar to get_batch; maxlen shoule be fixed as 50 (cnn_discriminator)
            while len(sentence) < maxlen:
                sentence = np.append(sentence, [[0]] , axis=0)
            if len(sentence) > maxlen:
                sentence = np.delete(sentence,np.s_[maxlen:],0)
                
            translated_sentences_.append(sentence)
        
        translated_sentences_ = np.array(translated_sentences_)
        translated_sentences_ = translated_sentences_.swapaxes(0,1)
        translated_sentences_shape = translated_sentences_.shape
        translated_sentences_ = translated_sentences_.reshape(translated_sentences_shape[0],translated_sentences_shape[1])
        
        batch = self.get_batch(data_values[0],data_values[2], label=None, maxlen = 50)
        batch[0] = np.concatenate([batch[0],batch[0]],axis=1)
        batch[1] = np.concatenate([batch[1],translated_sentences_],axis=1)
        
        label = []
        for i in range(0,int(batch[0].shape[1]/2)):
            label.append([1,0])
        for i in range(int(batch[0].shape[1]/2),batch[0].shape[1]):
            label.append([0,1])
        batch.append(np.array(label, dtype=FLOAT)) 
        
        return batch
    
    def prepare_data_not_MC(self, data_values, generator, maxlen=50):
        input_sentences, translated_sentences = generator.translate_multinomial(data_values, maxlen)
        # Khoa: Select a part of a sentence for cnn_discriminator
        translated_sentences_ = []
        for sentence in translated_sentences:
            random_index = np.random.randint(0,len(sentence),1)
            sentence = sentence[0:random_index[0]]
            # Khoa: Modify (Increase and decrease) the size of translated_sentences_ into sentence lenght = 50 for convolution
            # Similar to get_batch; maxlen shoule be fixed as 50 (cnn_discriminator)
            
            while len(sentence) < maxlen:
                sentence = np.append(sentence, [[0]] , axis=0)
            if len(sentence) > maxlen:
                sentence = np.delete(sentence,np.s_[maxlen:],0)
                
            translated_sentences_.append(sentence)
            
        translated_sentences_ = np.array(translated_sentences_)
        translated_sentences_ = translated_sentences_.swapaxes(0,1)
        translated_sentences_shape = translated_sentences_.shape
        translated_sentences_ = translated_sentences_.reshape(translated_sentences_shape[0],translated_sentences_shape[1])
        
        
        batch = self.get_batch(data_values[0],data_values[2], label=None, maxlen = 50)
        batch[0] = np.concatenate([batch[0],batch[0]],axis=1)
        batch[1] = np.concatenate([batch[1],translated_sentences_],axis=1)
        
        
        label = []
        for i in range(0,int(batch[0].shape[1]/2)):
            label.append([1,0])
        for i in range(int(batch[0].shape[1]/2),batch[0].shape[1]):
            label.append([0,1])
        batch.append(np.array(label, dtype=FLOAT)) 
        
        return batch
        
    # Khoa: Modify (Increase and decrease) the size of batch into sentence length = 50 for convolution
    def get_batch(self, x, y, label=None, maxlen=50):
        x = list(x)
        y = list(y)
                
        eos = np.zeros(np.array(x[0]).shape)
                
        while len(x) < maxlen:
            x = np.append(x, [eos] , axis=0)
        while len(y) < maxlen:
            y = np.append(y, [eos] , axis=0)
            
        if len(x) > maxlen:
            x = np.delete(x,np.s_[maxlen:],0)
            
        if len(y) > maxlen:
            y = np.delete(y,np.s_[maxlen:],0)
                    
        batch = []
        batch.append(np.array(x, dtype=INT))
        batch.append(np.array(y, dtype=INT))
        
        if label is not None:
            batch.append(np.array(label, dtype=FLOAT))
            
        return list(batch)

    def param_init_conv(self, params, filter_shape, scale='he', prefix='conv'):
        # input_shape : (input_channels, input_rows, input_cols)
        # filter_shape: (output_channels, input_channels, filter_rows, filter_cols)
        n_out_chan, n_inp_chan, n_filt_row, n_filt_col = filter_shape
    
        W = norm_weight(n_filt_row*n_filt_col*n_inp_chan, n_out_chan, scale=scale)
        # Conv layer weights as 4D tensor
        params[pp(prefix, 'W')] = W.reshape((n_out_chan, n_inp_chan, n_filt_row, n_filt_col))
        # 1 bias per output channel
        params[pp(prefix, 'b')] = np.zeros((n_out_chan, )).astype(FLOAT)
    
        return params
    

    def conv_layer(self, tparams, state_below, prefix='conv', activ='relu'):
        # state_below shape should be bc01
        return tensor.nnet.relu(tensor.nnet.conv2d(state_below, tparams[pp(prefix, 'W')], border_mode='valid') +
            tparams[pp(prefix, 'b')][None, :, None, None])
    
    # Khoa:
    def val_loss(self, mean=True):
        # True prediction
        prob_true = []

        for data in self.valid_iterator:
            batch_discriminator = self.get_batch(list(data.values())[0],list(data.values())[2],list(data.values())[4] )
            probs = self.get_probs_valid(*batch_discriminator)
            probs_len = len(probs)
            probs = np.array(probs)*np.array(list(data.values())[4])
            probs = probs.sum(1)
            true_num= sum(1 for prob in probs if prob > 0.5)
            prob_true.append(1 - (true_num/probs_len))
        if mean:
            return np.array(prob_true).mean()
        else:
            return np.array(prob_true)
    # Khoa.

    def concatenate(self, tensor_list, axis=0):
        """
        Alternative implementation of `theano.tensor.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.tensor.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.
        :usage:
            >>> x, y = theano.tensor.matrices('x', 'y')
            >>> c = concatenate([x, y], axis=1)
        :parameters:
            - tensor_list : list
                list of Theano tensor expressions that should be concatenated.
            - axis : int
                the tensors will be joined along this axis.
        :returns:
            - out : tensor
                the concatenated tensor expression.
        """
        concat_size = sum(tt.shape[axis] for tt in tensor_list)
    
        output_shape = ()
        for k in range(axis):
            output_shape += (tensor_list[0].shape[k],)
        output_shape += (concat_size,)
        for k in range(axis + 1, tensor_list[0].ndim):
            output_shape += (tensor_list[0].shape[k],)
    
        out = tensor.zeros(output_shape)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)
    
            out = tensor.set_subtensor(out[indices], tt)
            offset += tt.shape[axis]
    
        return out
    
class ClassificationIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None, **kwargs):
        super(ClassificationIterator, self).__init__(batch_size, seed, mask, shuffle_mode, logger)

        assert 'srcfile' in kwargs, "Missing argument srcfile"
        assert 'trgfile' in kwargs, "Missing argument trgfile"
        assert 'srcdict' in kwargs, "Missing argument srcdict"
        assert 'trgdict' in kwargs, "Missing argument trgdict"
        assert 'labelfile'   in kwargs, "Missing argument label"
        assert batch_size > 1, "Batch size should be > 1"

        self._print('Shuffle mode: %s' % shuffle_mode)

        self.srcfile = kwargs['srcfile']
        self.trgfile = kwargs['trgfile']
        self.srcdict = kwargs['srcdict']
        self.trgdict = kwargs['trgdict']
        self.labelfile   = kwargs['labelfile']

        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')


        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        self._keys.append(self.trg_name)
        if self.mask:
            self._keys.append("%s_mask" % self.trg_name)
            
        self._keys.append('label')

    def read(self):
        seqs = []
        sf = fopen(self.srcfile, 'r')
        tf = fopen(self.trgfile, 'r')
        lf = fopen(self.labelfile, 'r')

        for idx, (sline, tline, lline) in enumerate(zip(sf, tf, lf)):
            sline = sline.strip()
            tline = tline.strip()
            lline = lline.strip()

            # Exception if empty line found
            if sline == "" or tline == "" or lline == "":
                continue

            sseq = [self.srcdict.get(w, 1) for w in sline.split(' ')]
            tseq = [self.trgdict.get(w, 1) for w in tline.split(' ')]
            lseq = [int(w) for w in lline.split(' ')]
            
            # if given limit vocabulary
            if self.n_words_src > 0:
                sseq = [w if w < self.n_words_src else 1 for w in sseq]

            # if given limit vocabulary
            if self.n_words_trg > 0:
                tseq = [w if w < self.n_words_trg else 1 for w in tseq]
                

            # Append sequences to the list
            seqs.append((sseq, tseq,lseq))
        
        sf.close()
        tf.close()
        lf.close()

        # Save sequences
        self._seqs = seqs

        # Number of training samples
        self.n_samples = len(self._seqs)
        
        # Set batch processor function
        self._process_batch = (lambda idxs: self.mask_seqs(idxs))

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=1)
        else:
            self.rewind()

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


    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
        src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
        trg, trg_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
        label = [self._seqs[i][2] for i in idxs]
        return (src, src_mask, trg, trg_mask, label)
