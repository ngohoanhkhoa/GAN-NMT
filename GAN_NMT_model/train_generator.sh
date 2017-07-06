#PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
#export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
#export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/

export THEANO_FLAGS=device=cpu,floatX=float32
#export OMP_NUM_THREADS=8

python ~/Documents/Stage/nmtpy/nmtpy/train_generator.py -c ~/Documents/Stage/GAN_NMT_model/attention_en_fr.conf
