#!/bin/bash
ps_num=4
worker_num=4

ps_ips=('10.20.30.10' '10.20.30.11' '10.20.30.12' '10.20.30.13')
worker_ips=('10.20.30.100' '10.20.30.101' '10.20.30.102' '10.20.30.103')

echo "Building the host for ps and workers:"
ps_hosts=''
ps_port=2222
index=True
for ps_host in ${ps_ips[*]}
do
	if [ $index == True ];then ps_hosts=$ps_host:$ps_port;index=False
	else ps_hosts=$ps_hosts','$ps_host:$ps_port
	fi
done

echo $ps_hosts

worker_hosts=''
index=True
for worker_host in ${worker_ips[*]}
do
	if [ $index == True ];then worker_hosts=$worker_host:$ps_port;index=False
	else worker_hosts=$worker_hosts','$worker_host:$ps_port
	fi
done

ps_limitresources='--cpus=8 --memory=20G'
worker_limitresources='--cpus=5 --memory=10G'

scripts_dir=/home/h3cai01/wangfeicheng/Tensorflow
WORKDIR="/Tensorflow"

#run on imagenet
script=/Tensorflow/docker-multiple/ResNet/resnet_imagenet_main.py
data_dir=/home/hdd0/dataset/imagenet2012/ILSVRC2012
#data_dir=/home/ssd1/xiejx/imagenet/ILSVRC2012


#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-ckpt"
#log_dir
log_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-log"
#eval_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-eval"

train_steps=100001

ps0_cmd="python $script --train_data_path=$data_dir  --eval_data_path=$data_dir \  
--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir \
--batch_size 64 --num_gpus=1 --train_steps=$train_steps \
--ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" 
worker_cmd="python  $script --train_data_path $data_dir  --eval_data_path=$data_dir \  
--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir \
--batch_size 64 --num_gpus 1  --train_steps=$train_steps \
--ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" 


docker network create --driver=bridge --subnet=10.20.30.0/24 --gateway=10.20.30.1 tfdocker

echo "generating the contain for ps and worker:"
for index in $(seq $ps_num)
do
	index=`expr $index - 1`
	echo "generating the ps:$index"
	nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $ps_limitresources  \
	--net tfdocker --ip ${ps_ips[$index]} \
	-e "CUDA_VISIBLE_DEVICES="  --name tfps$index ufoym/deepo:all-jupyter-py36
	
	echo "executing the ps:$index command"	
	docker exec --workdir $WORKDIR -d tfps$index $ps0_cmd --job_name=ps --task_index=$index
done

for index in $(seq $worker_num)
do
	index=`expr $index - 1`
    echo "generating the worker:$index"
	nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $worker_limitresources \
	--net tfdocker --ip ${worker_ips[$index]} \
	-e "CUDA_VISIBLE_DEVICES=$index" --name tfworker$index ufoym/deepo:all-jupyter-py36
	
	echo "executing the worker:$index command"	
	if [ $index == 0 ]
	then
		docker exec --workdir $WORKDIR -i tfworker$index $worker_cmd --job_name=worker --task_index=$index >resnet_imagenet_train_log.log 2>&1 &
	else
		docker exec --workdir $WORKDIR -d tfworker$index $worker_cmd --job_name=worker --task_index=$index
	fi
done


