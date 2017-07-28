#PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
#export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
#export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/

export THEANO_FLAGS=device=cpu,floatX=float32
#export OMP_NUM_THREADS=8

#nmt-translate -o ~/Documents/Stage/GAN_NMT_model/data_test/train.fr -S ~/Documents/Stage/GAN_NMT_model/data_test/train.en -m ~/Documents/Stage/GAN_NMT_model/data_test/attention-e512-r1000-adadelta_1e+00-bs3-bleu-each5_do_d0.0-gc1-init_xavier-s1234.1-val001-bleu_0.000.npz

python ~/Documents/Stage/nmtpy/nmtpy/nmt-translate.py -D argmax -o ~/Documents/Stage/GAN_NMT_model/data_translate/out_train.fr -S ~/Documents/Stage/GAN_NMT_model/data_translate/train.en -m ~/Documents/Stage/GAN_NMT_model/data_translate/attention-e512-r1024-adadelta_1e+00-bs80-bleu-each5000_do_d0.0-gc1-init_xavier-s1234.1-val001-bleu_38.850.npz