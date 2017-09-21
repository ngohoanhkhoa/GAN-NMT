export THEANO_FLAGS=device=cpu

python ../nmtpy/nmtpy/GAN/train_rnnlm.py -c model_conf/rnnlm_fr.conf
