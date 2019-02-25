#!/bin/bash
scripts_dir=/home/h3cai01/wangfeicheng/Tensorflow
WORKDIR="/Tensorflow"
script="/Tensorflow/docker-multiple/ResNet/resnet_imagenet_eval.py"
data_dir=/home/hdd0/dataset/imagenet2012/ILSVRC2012

#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-ckpt"
#log_dir
log_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-log"
#eval_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-eval"


eval_cmd="python  $script --eval_data_path $data_dir --train_dir=$train_dir --eval_dir=$eval_dir \
--num_gpus 1 --mode=eval"

nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir -e "CUDA_VISIBLE_DEVICES=7" --name tf-eval ufoym/deepo:all-jupyter-py36

echo "the train dir is "$train_dir
docker exec --workdir $WORKDIR -i tf-eval $eval_cmd