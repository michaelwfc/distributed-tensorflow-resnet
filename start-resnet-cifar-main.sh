#!/bin/bash   
  
#training with distributed mode and evalation with another process
mode='train'
variable_update=parameter_server
# horovod
train_steps=200001


#run on cifar
#script="/Tensorflow/docker-multiple/ResNet/resnet_cifar_main.py"
script="/Tensorflow/docker-multiple/ResNet/resnet_cifar_train.py"
#eval on cifar
eval_script="/Tensorflow/docker-multiple/ResNet/resnet_cifar_main.py"
data_dir=/home/wangfeicheng/Tensorflow/cifar10-tensorflow/data
train_data_path=/home/wangfeicheng/Tensorflow/cifar10-tensorflow/data
eval_data_path=/home/wangfeicheng/Tensorflow/cifar10-tensorflow/data/cifar-10-batches-bin/test_batch.bin


if [ $variable_update = 'parameter_server' ];then
	#image=ufoym/deepo:all-jupyter-py36
	image=3.2.1.12/tensorflow-1.12-gpu-py3:1.12
	#checkpoint_dir
	train_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-ckpt"
	#log_dir
	log_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-log/train"
	#eval_dir
	eval_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-log/validation"
elif [ $variable_update = 'horovod' ];then
	image=horovod_resnet50:v2
	#checkpoint_dir
	train_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-horovod-ckpt"
	#log_dir
	log_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-horovod-log/train"
	#eval_dir
	eval_dir="/Tensorflow/docker-multiple/ResNet/test/resnet50-cifar-horovod-log/validation"
	#horovod setting
	worker_num=4
	gpu_num=2
	gpu_per_worker=1
fi
	
###==========================parameter_server===========================
###=====================================================================
###=====================================================================

if [ $variable_update = 'parameter_server' ];then
	echo "Starting the variable_update:$variable_update"
	ps_num=2
	worker_num=2
	ps_ips=('10.20.30.10' '10.20.30.11' '10.20.30.12' '10.20.30.13')
	worker_ips=('10.20.30.100' '10.20.30.101' '10.20.30.102' '10.20.30.103')
	echo "Building the host for ps and workers:"
	ps_hosts=''
	ps_port=2222
	count=0
	index=True
	for ps_host in ${ps_ips[*]}
	do
		if [ $count -lt $ps_num ];then
			if [ $index == True ];then ps_hosts=$ps_host:$ps_port;index=False
			else ps_hosts=$ps_hosts','$ps_host:$ps_port
			fi
		fi
		count=`expr $count + 1`
	done
	
	echo $ps_hosts
	worker_hosts=''
	count=0
	index=True
	for worker_host in ${worker_ips[*]}
	do
		if [ $count -lt $worker_num ];then
			if [ $index == True ];then worker_hosts=$worker_host:$ps_port;index=False
			else worker_hosts=$worker_hosts','$worker_host:$ps_port
			fi
		fi
		count=`expr $count + 1`
		
	done
	echo $worker_host
	
	ps_limitresources='--cpus=8 --memory=20G'
	worker_limitresources='--cpus=5 --memory=10G'

	scripts_dir=/home/wangfeicheng/Tensorflow
	WORKDIR="/Tensorflow"

	ps0_cmd="python $script --train_data_path=$data_dir  --eval_data_path=$data_dir \  
	--train_dir=$train_dir  --log_dir=$log_dir --mode=$mode \
	--batch_size 64 --num_gpus=0 --train_steps=$train_steps \
	--ps_hosts=$ps_hosts --worker_hosts=$worker_hosts \
	--variable_update=$variable_update"
	worker_cmd="python  $script --train_data_path $data_dir  --eval_data_path=$data_dir \  
	--train_dir=$train_dir --eval_dir=$eval_dir --log_dir=$log_dir  --mode=$mode \
	--batch_size 64 --num_gpus 1  --train_steps=$train_steps \
	--ps_hosts=$ps_hosts --worker_hosts=$worker_hosts \ 
	--variable_update=$variable_update" 

	docker network create --driver=bridge --subnet=10.20.30.0/24 --gateway=10.20.30.1 tfdocker

	echo -e "Training:\n generating the contain for ps and worker:"
	for index in $(seq $ps_num)
	do
		index=`expr $ps_num - $index`
		echo "generating the ps:$index"
		nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $ps_limitresources  \
		--net tfdocker --ip ${ps_ips[$index]} \
		-e "CUDA_VISIBLE_DEVICES="  --name tfps$index $image
		
		echo "executing the ps:$index command"	
		docker exec --workdir $WORKDIR -d tfps$index $ps0_cmd --job_name=ps --task_index=$index
	done

	for index in $(seq $worker_num)
	do
		index=`expr $worker_num - $index`
		gpu_index=`expr $index + 6`
		echo "generating the worker:$index"
		nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir $worker_limitresources \
		--net tfdocker --ip ${worker_ips[$index]} \
		-e "CUDA_VISIBLE_DEVICES=$gpu_index" --name tfworker$index $image
		
		echo "executing the worker:$index command"	
		if [ $index == 0 ]
		then
			docker exec --workdir $WORKDIR -i tfworker$index $worker_cmd --job_name=worker --task_index=$index &
			#>resnet_imagenet_main_20190226.log 2>&1 &
		else
			docker exec --workdir $WORKDIR -d tfworker$index $worker_cmd --job_name=worker --task_index=$index
		fi
	done
	
###==========================horovod====================================
###=====================================================================
###=====================================================================
elif [ $variable_update = 'horovod' ];then
	echo "Starting the variable_update:$variable_update"
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
		-e "CUDA_VISIBLE_DEVICES=$index" --name $worker_name$index $image
	done

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

	#运行脚本
	#master:
	master_cmd="mpirun -np $gpu_num -H $worker_hosts -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
	-x LD_LIBRARY_PATH -x PATH -mca btl_tcp_if_exclude docker0,lo -x NCCL_SOCKET_IFNAME=^docker0,lo \
	-mca pml ob1 -mca btl ^openib -mca plm_rsh_args '-p 12345' \
	python $script --train_data_path=$train_data_path  --eval_data_path=$eval_data_path \
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
			docker exec --workdir $WORKDIR -i $worker_name$index bash -c "$master_cmd" &
			#>resnet_imagenet_main_20190226.log 2>&1 &
		else
			echo "executing the worker:$index command"
			docker exec -d --workdir $WORKDIR  $worker_name$index bash -c "$worker_cmd" 
		fi
	done
else
	echo "Do nothing"
fi

echo -e "Validation:"

eval_cmd="python  $eval_script --eval_data_path $eval_data_path --train_dir=$train_dir --eval_dir=$eval_dir \
--num_gpus 0 --mode=eval"

nvidia-docker run -t -d -v $scripts_dir:$WORKDIR -v $data_dir:$data_dir -e "CUDA_VISIBLE_DEVICES=7" --name tf-eval $image
docker exec --workdir $WORKDIR -i tf-eval $eval_cmd
