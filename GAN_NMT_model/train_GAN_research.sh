#PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
#export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
#export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/

export THEANO_FLAGS=device=cpu,floatX=float32

python ~/Documents/Stage/nmtpy/nmtpy/train_GAN_MC.py -c ~/Documents/Stage/GAN_NMT_model/model_conf/nmt_GAN_en_fr_research.conf