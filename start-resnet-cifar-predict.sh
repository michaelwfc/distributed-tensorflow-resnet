#!/bin/bash
#run on cifar
scripts_dir=/home/h3cai01/wangfeicheng/Tensorflow
WORKDIR="/Tensorflow"
script="/Tensorflow/docker-multiple/ResNet/resnet_cifar_predict.py"
data_dir=/home/hdd0/dataset/cifar10_data
train_data_path=/home/hdd0/dataset/cifar10_data
eval_data_path=/home/hdd0/dataset/cifar10_data/cifar-10-batches-bin/test_batch.bin

#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-ckpt-20190218/model.ckpt-107738"
#summary_dir
log_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-log"
#eval_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-eval8" 

predict_cmd="python  $script --eval_data_path $eval_data_path --train_dir=$train_dir --eval_dir=$eval_dir \
--num_gpus 0 --mode=eval"

nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir -e "CUDA_VISIBLE_DEVICES=" --name tf-eval ufoym/deepo:all-jupyter-py36

docker exec --workdir $WORKDIR -i tf-eval $predict_cmd
