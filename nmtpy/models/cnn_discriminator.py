# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as tensor
from ..layers import tanh, get_new_layer
from ..defaults import INT, FLOAT
from ..nmtutils import load_dictionary, norm_weight, ortho_weight, pp
from ..iterators.text import TextIterator


from .basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Load dictionaries
        dicts = kwargs['dicts']

        # Let's default to GRU
        self.rnn_type = kwargs.get('rnn_type', 'gru')

        self.src_dict, src_idict = load_dictionary(dicts['src'])
        self.n_words = min(self.n_words, len(self.src_dict)) \
                if self.n_words > 0 else len(self.src_dict)

        self.set_options(self.__dict__)
        self.src_idict = src_idict
        self.set_trng(seed)
        self.set_dropout(False)
        self.logger = logger

    def load_valid_data(self):
        self.valid_iterator = TextIterator(
                                batch_size=1,
                                mask=True,
                                shuffle_mode=None,
                                file=self.data['valid_src'],
                                dict=self.src_dict,
                                n_words=self.n_words,
                                name='y') # This is important for the loss to be correctly normalized!
        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = TextIterator(
                                batch_size=self.batch_size,
                                mask=True,
                                shuffle_mode=None, # or simple or trglen, not tested in rnnlm.
                                file=self.data['train_src'],
                                dict=self.src_dict,
                                n_words=self.n_words)

        self.train_iterator.read()
        self.load_valid_data()

    def init_params(self):
        params = OrderedDict()
        
        # embedding weights for encoder
        params['Wemb_enc_x'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_enc_y'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        # input_shape : (input_channels, input_rows, input_cols)
        # filter_shape: (output_channels, input_channels, filter_rows, filter_cols)
        # Check layers.py and nmtutils.py
        input_shape = (1,4,4)
        filter_shape = (1,1,4,4)
        
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_x')
        params = get_new_layer('conv')[0](params, input_shape, filter_shape, scale='he', prefix='conv_y')
        
        params = get_new_layer('ff')[0](params, prefix='ff_logit', nin=self.out_emb_dim, nout=2)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)
        
        label = tensor.vector('label', dtype=INT)

        # Store tensors
        self.inputs = OrderedDict()
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask
        self.inputs['y']        = y         # Target words
        self.inputs['y_mask']   = y_mask    # Target mask
        
        self.inputs['label']    = label     # Label of sentence

        n_timesteps = x.shape[0]
        n_timesteps_trg = y.shape[0]
        n_samples = x.shape[1]

        # input x word embedding
        emb_x = self.tparams['Wemb_enc_x'][x.flatten()]
        emb_x = emb_x.reshape([n_timesteps, n_samples, self.embedding_dim])
        
        # input y word embedding
        emb_y = self.tparams['Wemb_enc_y'][y.flatten()]
        emb_y = emb_y.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        
        # Convolutional operation on x, y
        proj_x = conv_layer(self.tparams, emb_x, prefix='conv_x', mask=x_mask)
        
        proj_y = conv_layer(self.tparams, emb_y, prefix='conv_y', mask=y_mask)
        
        conv_pool_size = [5,5]
        # Max-over-time pooling operation over the feature maps
        if conv_pool_size is not None:
        # Pool each feature map individually, using maxpooling
            feature_pool_x = tensor.signal.pool.pool_2d(input=proj_x, ds=conv_pool_size, ignore_border=True)
            feature_pool_y = tensor.signal.pool.pool_2d(input=proj_y, ds=conv_pool_size, ignore_border=True)
            
        # Concatenate x and y
        x_y = tensor.concatenate([feature_pool_x[0], feature_pool_y[0]], axis=0)
        
        # Get probability from feedforward layer
        logit = get_new_layer('ff')[1](self.tparams, x_y, prefix='ff_logit', activ='tanh')
        
        log_probs = -tensor.nnet.logsoftmax(logit)
        
        # cost
        #f_log_probs_detailled return the log probs array correponding to each word log probs
        self.f_log_probs_detailled = theano.function(list(self.inputs.values()), cost)
        cost = (cost * x_mask).sum(0)

        cost = label - log_probs
        cost = cost.reshape([n_samples])
        cost = cost.sum(0)
        
        self.f_log_probs = theano.function(list(self.inputs.values()), cost)
    
        return cost

## weight initializer, normal by default
#def norm_weight(nin, nout, scale=0.01, ortho=True):
#    if scale == "xavier":
#        # Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."
#        # International conference on artificial intelligence and statistics. 2010.
#        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
#        scale = 1. / np.sqrt(nin)
#    elif scale == "he":
#        # Claimed necessary for ReLU
#        # Kaiming He et al. (2015)
#        # Delving deep into rectifiers: Surpassing human-level performance on
#        # imagenet classification. arXiv preprint arXiv:1502.01852.
#        scale = 1. / np.sqrt(nin/2.)
#
#    if nout == nin and ortho:
#        W = ortho_weight(nin)
#    else:
#        W = scale * np.random.randn(nin, nout)
#    return W.astype(FLOAT)

#def norm_conv_weight(filter_shape,scale=0.01, ortho=False):
#    # TODO:write the ortho_weight function for the convolutional weights
#    W = scale * numpy.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3])
#    return W.astype('float32')
#
#def param_init_conv(params, input_shape, filter_shape, scale='he', prefix='conv'):
#    # input_shape : (input_channels, input_rows, input_cols)
#    # filter_shape: (output_channels, input_channels, filter_rows, filter_cols)
#    n_inp_chan, n_inp_row, n_in_col = input_shape
#    n_out_chan, n_inp_chan, n_filt_row, n_filt_col = filter_shape
#
#    W = norm_weight(n_filt_row*n_filt_col*n_inp_chan, n_out_chan, scale=scale)
#    # Conv layer weights as 4D tensor
#    params[pp(prefix, 'W')] = W.reshape((n_out_chan, n_inp_chan, n_filt_row, n_filt_col))
#    # 1 bias per output channel
#    params[pp(prefix, 'b')] = np.zeros((n_out_chan, )).astype(FLOAT)
#
#    return params

def conv_layer(tparams, state_below, prefix='conv', activ='relu'):
    # state_below shape should be bc01
    return eval(activ) (
        tensor.nnet.conv2d(state_below, tparams[pp(prefix, 'W')], border_mode='valid') +
        tparams[pp(prefix, 'b')][None, :, None, None]
        )

#def conv_layer(tparams,sentences, options,prefix='conv',
#               mask=None,
#               emb_dropout=None,
#               rec_dropout=None,
#               profile=False,
#               **kwargs):
#    '''
#    Feature Maps: FM
#    :type sentences: theano.tensor.dtensor3
#    :param sentences: symbolic tensor, of shape image_shape(input_shape) (#samples,1,#timesteps,dim_word) 
#                    
#    :type filter_shape: tuple or list of length 4
#    :param filter_shape: (#FM at layer m,#input FM or #FM at layer m-1,filter height, filter width)
#
#    :type image_shape: tuple or list of length 4
#    :param image_shape: (batch size,#input FM,image height, image width) it is the deprecated alias of input_shape
#
#    :type poolsize: tuple or list of length 2
#    :param poolsize: the downsampling (pooling) factor (#rows, #cols)
#
#    :type conv_out: output of the convolution operation 
#    :param conv_out:a tensor of size (batch size, output channels, output rows, output columns)
#    '''
#    # mask
#    if mask is None:
#        mask = tensor.alloc(1., sentences.shape[2]-options['conv_filter_shape'][2]+1, 1)
#
#    conv_out = conv2d(input=sentences,filters=tparams[pp(prefix, 'W')],
#                      filter_shape=options['conv_filter_shape']) 
#    #theano.printing.Print('mask', ["shape"])(mask) #debugging
#    #filter_shape_ = options['conv_filter_shape'][2]-1
#    #print filter_shape_
#    #filter_shape_ = theano.printing.Print('filter_shape_')(filter_shape_)
#    
#    #mask_ = mask[options['conv_filter_shape'][2]-1:,:]
#    #mask_ind = (mask_[0,:]<1).nonzero()
#    #mask_ = theano.tensor.set_subtensor(mask_[0,mask_ind], 1)
#    #mask_trans = mask_.T
#    conv_out_ = mask.reshape((mask.shape[1],1,mask.shape[0],1),ndim=4)*conv_out
#
#    
#    if options['conv_pool_size'] is not None:
#        # pool each feature map individually, using maxpooling
#        conv_out_ = tensor.signal.pool.pool_2d(
#            input=conv_out_,
#            ds=options['conv_pool_size'],
#            ignore_border=True)
#        
#    # add the bias term. Since the bias is a vector (1D array), we first
#    # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
#    # thus be broadcasted across mini-batches and feature map width & height
#    output = tensor.tanh(conv_out_ + tparams[pp(prefix, 'b')].dimshuffle('x', 0, 'x', 'x'))
#
#    
#    return output
