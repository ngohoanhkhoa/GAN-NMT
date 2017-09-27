export THEANO_FLAGS=device=cpu,floatX=float32

python ../nmtpy/nmtpy/GAN/train_GAN_MC.py -c model_conf/nmt_GAN_en_fr_research.conf