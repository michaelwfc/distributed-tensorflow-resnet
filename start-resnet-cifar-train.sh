#!/bin/bash
ps0ip='10.20.30.10'
worker0ip='10.20.30.100'
worker1ip='10.20.30.101'
worker2ip='10.20.30.102'
worker3ip='10.20.30.103'

ps0ip_cmd='--net tfdocker --ip 10.20.30.10'
worker0ip_cmd='--net tfdocker --ip 10.20.30.100'
worker1ip_cmd='--net tfdocker --ip 10.20.30.101'
worker2ip_cmd='--net tfdocker --ip 10.20.30.102'
#worker3ip_cmd='--net tfdocker --ip 10.20.30.103'
#worker4ip_cmd='--net tfdocker --ip 10.20.30.104'
limitresources='--cpus=8 --memory=50G'

scripts_dir=/home/h3cai01/wangfeicheng/Tensorflow
WORKDIR="/Tensorflow"

#run on cifar
script="/Tensorflow/docker-multiple/ResNet/resnet_cifar_main.py"
data_dir=/home/hdd0/dataset/cifar10_data
train_data_path=/home/hdd0/dataset/cifar10_data
eval_data_path=/home/hdd0/dataset/cifar10_data/cifar-10-batches-bin/test_batch.bin

#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-ckpt"
#log_dir
log_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-log"
#eval_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-cifar-eval"

train_steps=100001

ps0_cmd="python $script --train_data_path=$train_data_path  --eval_data_path=$eval_data_path \  
--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir \
--batch_size 64 --num_gpus=0 --train_steps=$train_steps \
--ps_hosts=10.20.30.10:2222 --worker_hosts=10.20.30.100:2222,10.20.30.101:2222"
#,10.20.30.102:2222"
#,10.20.30.103:2222" #10.20.30.104:2222
worker_cmd="python  $script --train_data_path $train_data_path  --eval_data_path=$eval_data_path \  
--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir \
--batch_size 64 --num_gpus 1  --train_steps=$train_steps \
--ps_hosts=10.20.30.10:2222 --worker_hosts=10.20.30.100:2222,10.20.30.101:2222"
#,10.20.30.102:2222"
#,10.20.30.103:2222" #10.20.30.104:2222


docker network create --driver=bridge --subnet=10.20.30.0/24 --gateway=10.20.30.1 tfdocker
nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $limitresources $worker0ip_cmd -e "CUDA_VISIBLE_DEVICES=6" --name tfworker0 ufoym/deepo:all-jupyter-py36
nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $limitresources $worker1ip_cmd -e "CUDA_VISIBLE_DEVICES=7" --name tfworker1 ufoym/deepo:all-jupyter-py36
#nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $limitresources $worker2ip_cmd -e "CUDA_VISIBLE_DEVICES=6" --name tfworker2 ufoym/deepo:all-jupyter-py36
#nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $limitresources $worker0ip_cmd -e "CUDA_VISIBLE_DEVICES=0,1" --name tfworker0 ufoym/deepo:all-jupyter-py36

nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $limitresources $ps0ip_cmd  -e "CUDA_VISIBLE_DEVICES=7"  --name tfps0 ufoym/deepo:all-jupyter-py36

docker exec --workdir $WORKDIR -d tfps0 $ps0_cmd --job_name=ps --task_index=0
docker exec --workdir $WORKDIR -d tfworker1 $worker_cmd --job_name=worker --task_index=1
#docker exec --workdir $WORKDIR -d tfworker2 $worker_cmd --job_name=worker --task_index=2
#docker exec --workdir $WORKDIR -d tfworker3 $worker_cmd --job_name=worker --task_index=3
docker exec --workdir $WORKDIR -i tfworker0 $worker_cmd --job_name=worker --task_index=0
