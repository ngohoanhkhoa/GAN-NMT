export THEANO_FLAGS=device=cpu,floatX=float32


python ../nmtpy/nmtpy/GAN/train_generator.py -c model_conf/attention_en_fr.conf
