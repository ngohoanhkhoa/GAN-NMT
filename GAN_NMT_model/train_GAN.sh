#PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
#export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
#export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/

export THEANO_FLAGS=device=cpu,floatX=float32
#export OMP_NUM_THREADS=8

#python ~/Documents/Stage/nmtpy/setup.py install
#python ~/Documents/Stage/nmtpy/setup.py dev

#nmt-train -c ~/Documents/Stage/GAN_NMT_model/attention_en-fr.conf

#python ~/Documents/Stage/nmtpy/nmtpy/train_discriminator.py -c ~/Documents/Stage/GAN_NMT_model/discriminator_en_fr.conf

#python ~/Documents/Stage/nmtpy/nmtpy/train_generator.py -c ~/Documents/Stage/GAN_NMT_model/attention_en_fr.conf

python ~/Documents/Stage/nmtpy/nmtpy/train_GAN.py -c ~/Documents/Stage/GAN_NMT_model/nmt_GAN_en_fr.conf

#python ~/Documents/Stage/nmtpy/nmtpy/train_GAN.py -c ~/Documents/Stage/GAN_NMT_model/nmt_GAN_en_fr.conf -i ~/Documents/Stage/GAN_NMT_model/models/attention_en_fr/attention-e512-r1000-adadelta_1e+00-bs3-bleu-each5_do_d0.0-gc1-init_xavier-s1234.1-val001-bleu_0.000.npz -id ~/Documents/Stage/GAN_NMT_model/models/discriminator_en_fr/cnn_discriminator-e512-adadelta_1e+00-bs2-bleu-each5-gc1-init_xavier-s1234.1-val008-loss_0.127.npz