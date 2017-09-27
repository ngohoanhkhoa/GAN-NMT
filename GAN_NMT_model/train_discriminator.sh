export THEANO_FLAGS=device=cpu,floatX=float32

python ../nmtpy/nmtpy/GAN/train_discriminator.py -c model_conf/discriminator_en_fr.conf