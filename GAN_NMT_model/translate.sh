#PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
#export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
#export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/

export THEANO_FLAGS=device=cpu,floatX=float32

python ../nmtpy/nmtpy/GAN/nmt-translate.py -D sample -o ~/Documents/Stage/GAN_NMT_model/data_translate/out_train_multinomial.fr -S ~/Documents/Stage/GAN_NMT_model/data_translate/train.en -m ~/Documents/Stage/GAN_NMT_model/data_translate/attention-e512-r1024-adadelta_1e+00-bs80-bleu-each5000_do_d0.0-gc1-init_xavier-s1234.1-val001-bleu_38.850.npz