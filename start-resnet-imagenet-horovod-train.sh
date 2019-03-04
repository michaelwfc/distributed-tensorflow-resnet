#!/bin/bash  

mode='train'
train_steps=100001  
worker_num=4
gpu_num=4
gpu_per_worker=1  

worker_ips=('10.20.40.100' '10.20.40.101' '10.20.40.102' '10.20.40.103')
worker_hosts=''
index=True
for worker_host in ${worker_ips[*]}
do
	if [ $index == True ];then worker_hosts=$worker_host:$gpu_per_worker;index=False
	else worker_hosts=$worker_hosts','$worker_host:$gpu_per_worker
	fi
done

echo "The worker_hosts:"$worker_hosts




worker_name=horovod_worker

docker_network=horovod_bridge
echo "Buliding the docker_network:$docker_network"
docker network create --driver=bridge --subnet=10.20.40.0/24 --gateway=10.20.40.1 -o parent=enp24s0d1  $docker_network
#docker network create  -d macvlan --subnet=10.20.40.0/24 --gateway=10.20.40.1  -o parent=enp24s0d1 $docker_network

scripts_dir=/home/h3cai01/wangfeicheng/Tensorflow
WORKDIR="/Tensorflow"
data_dir=/home/hdd0/dataset
worker_limitresources='--cpus=10 --memory=40G'


echo "Generating the contain for worker:"
for index in $(seq $worker_num)
do
	index=`expr $worker_num - $index`
    echo "generating the worker:$index"
	nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $worker_limitresources \
	--network $docker_network --ip ${worker_ips[$index]} \
	-e "CUDA_VISIBLE_DEVICES=$index" --name $worker_name$index horovod_resnet50:v2
done

cuda_cmd="echo export CUDA_VISIBLE_DEVICES=$index>>/etc/profile;source /etc/profile;echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

'''
echo "Generating the CUDA_VISIBLE_DEVICES for worker:"
for index in $(seq $worker_num)
do
	index=`expr $worker_num - $index`
	echo "generating the CUDA_VISIBLE_DEVICES for the worker:$index command"	
	docker exec --workdir $WORKDIR -id $worker_name$index bash -c "$cuda_cmd" 
done
'''

file="/root/.ssh/id_rsa.pub"
authorized_keys=/home/h3cai01/wangfeicheng/Tensorflow/docker-multiple/ResNet/authorized_keys

for index in $(seq $worker_num)
do	
	index=`expr $index - 1`
	echo "===============running the horovod_worker$index============"
	#启动ssh服务
	docker exec -it $worker_name$index bash -c "service ssh start"

	#自动化生成秘钥
	file_exist=$(docker exec -it $worker_name$index bash -c "ls -al /root/|grep .ssh|wc -l")
	#因为输出含有长度为2的字符，echo ${#file_exist},故截取第一个字符
	file_exist=${file_exist::1}
	
	
	if [ $file_exist = 0 ]; then
		echo "file not exist: $file_exist"
		echo "generating $file in $worker_name$index"
		keygen_cmd="echo -e '\n'|ssh-keygen -t rsa -N ''"
		docker exec -it $worker_name$index bash -c "$keygen_cmd"
	else
		echo "file  exist: $file_exist"
		echo "overwriting $file in $worker_name$index"
		keygen_cmd="echo -e '\ny'|ssh-keygen -t rsa -N ''"
		docker exec -it $worker_name$index bash -c "$keygen_cmd"	
	fi

	#将id_rsa.pub加入到host的文件下 authorized_keys
	echo -e "cat the key to $authorized_keys\n\n"
	if [ $index == 0 ];then
		docker exec -it $worker_name$index bash -c "cat $file" >$authorized_keys
	else
		docker exec -it $worker_name$index bash -c "cat $file">>$authorized_keys
	fi
done

for index in $(seq $worker_num)
do
	index=`expr $index - 1`
	echo "Copying $authorized_keys to $worker_name$index:/root/.ssh/"
	docker cp $authorized_keys $worker_name$index:/root/.ssh/
done

#run on cifar
script=/Tensorflow/docker-multiple/ResNet/resnet_imagenet_train_horovod.py
data_dir=/home/hdd0/dataset/imagenet2012/ILSVRC2012
 

#checkpoint_dir
train_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-horovod-ckpt"
#log_dir
log_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-horovod-log"
#eval_dir
eval_dir="/Tensorflow/docker-multiple/ResNet/resnet50-imagenet-horovod-eval"

variable_update=horovod
#运行脚本
#master:
master_cmd="mpirun -np $gpu_num -H $worker_hosts -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH -x PATH -mca btl_tcp_if_exclude docker0,lo -x NCCL_SOCKET_IFNAME=^docker0,lo \
-mca pml ob1 -mca btl ^openib -mca plm_rsh_args '-p 12345' \
python $script --train_data_path=$data_dir  --eval_data_path=$data_dir \
--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir --batch_size 64 --train_steps=$train_steps  \
--variable_update=$variable_update"
#worker:
worker_cmd="/usr/sbin/sshd -p 12345; sleep infinity"


echo "Executing the cmd for eachworker:"
for index in $(seq $worker_num)
do
	index=`expr $worker_num - $index`
	if [ $index == 0 ];	then
		echo "executing the master:$index command"	
		docker exec --workdir $WORKDIR -i $worker_name$index bash -c "$master_cmd" 
		#>resnet_imagenet_main_20190226.log 2>&1 &
	else
		echo "executing the worker:$index command"
		docker exec -d --workdir $WORKDIR  $worker_name$index bash -c "$worker_cmd" 
	fi
done