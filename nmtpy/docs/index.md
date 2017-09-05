Introduction
------------

**nmtpy** is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
for training neural machine translation models using Theano.

The basic motivation behind forking **dl4mt-tutorial** was to create a framework where it would be
easy to implement a new model by *merely* creating a new Python file.

Features
-----

### General
  - No shell script, everything is in Python 
  - Overhaul object-oriented refactoring of the code: clear separation of API and scripts that interface with the API
  - INI style configuration files to define everything regarding a training experiment
  - Transparent cleanup mechanism to kill stale processes, remove temporary files
  - Simultaneous logging of training details to stdout and log file
  
  - Supports out-of-the-box BLEU, METEOR and COCO eval metrics
  - Includes [subword-nmt](https://github.com/rsennrich/subword-nmt) utilities for training and applying BPE model
  - Plugin-like text filters for hypothesis post-processing (Example: BPE, Compound, Desegment)
  - Early-stopping and checkpointing based on perplexity, BLEU or METEOR
  - Ability to add new metrics easily
  - Single `.npz` file to store everything about a training experiment
  - Automatic free GPU selection and reservation using `nvidia-smi`
  - Shuffling between epochs
    - Simple shuffle
    - [Homogeneous batches of same-length samples](https://github.com/kelvinxu/arctic-captions) to improve training speed
  - Improved parallel translation decoding on CPU
  - Forced decoding i.e. rescoring using NMT
  - Export decoding informations into `json` for further visualization of attention weights
  
### Training
  - Improved numerical stability and reproducibility
  - Glorot/Xavier, He, Orthogonal weight initializations
  - Efficient SGD, Adadelta, RMSProp and ADAM
    - Single forward/backward theano function without intermediate variables
  - Initialization of a model with weights from another nmtpy model
    - Ability to freeze pre-trained weights
  - Several recurrent blocks:
    - GRU, Conditional GRU (CGRU) and LSTM
    - Multimodal attentive CGRU variants
  - [Layer Normalization](https://github.com/ryankiros/layer-norm) support for GRU
  - [Tied target embeddings](https://arxiv.org/abs/1608.05859)
  - Simple/Non-recurrent Dropout, L2 weight decay
  - Training and validation loss normalization for comparable perplexities
