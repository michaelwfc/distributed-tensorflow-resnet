#!/bin/bash
host=0
host_ips=('3.10.19.10' '3.10.19.11')
SCRIPT="/Tensorflow/docker-multiple/ResNet/resnet_cifar_main.py"
data_dir="/Tensorflow/docker-multiple/cifar10_data/"
#CKPT_DIR="$SCRIPT_DIR/$MODEL-ckpt"
CLUSTER_SPEC="--ps_hosts=${host_ips[0]}:5000,${host_ips[1]}:5000  \
--worker_hosts=${host_ips[0]}:5001,${host_ips[1]}:5001"
#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-ckpt"
#log_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-log" 


echo executing the distributed commands on host$host :ps$host 
CUDA_VISIBLE_DEVICES='0,1,2,3' python $SCRIPT \
--train_data_path $data_dir  --batch_size 64 --num_gpus 0  --train_steps 100 \
--job_name=ps $CLUSTER_SPEC  --task_index=$host \
--train_dir=$train_dir  --eval_dir=$eval_dir &

echo executing the distributed commands on host$host :worker$host
CUDA_VISIBLE_DEVICES='4,5,6,7'  python $SCRIPT \
--train_data_path $data_dir  --batch_size 64 --num_gpus 4  --train_steps 100 \
--job_name=worker $CLUSTER_SPEC  --task_index=$host --data_dir=$DATA_DIR \
--train_dir=$train_dir --eval_dir=$eval_dir
