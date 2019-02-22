#!/bin/bash
#################### 1.使用Dockerfile制作安装好ssh的image
#docker build -t my_tfps:v1 .


#################### 2.分别在两台机器上创建容器 macvlan 网络
#docker network create  -d macvlan --gateway 3.0.0.1 --subnet 3.0.0.0/8  -o parent=enp24s0 mydocker
#注: parent 参数可通过ifconfig查看



#################### 3.定义各host上的容器环境
#define the ip for 2 ps 
host_ips=('3.10.19.10' '3.10.19.11')
#define the port for ps and worker on each host
ports=('5000','5001')

#define the network for the ps and work in docker
host=0            #host编号
image=my_tfps:v1  #image名称
docker_connect=mydocker #docker网络名称
host_ip_cmd="--network $docker_connect --ip ${host_ips[$host]}"  
limitresources='--cpus=10 --memory=100G' #docker资源限制

#设置数据存储路径和脚本路径
SCRIPT_DIR="/home/h3cai01/wangfeicheng/Tensorflow"
#DATA_DIR="/home/hdd0/dataset/imagenet2012/ILSVRC2012"
DATA_DIR="/mnt/ainfs"
WORKDIR="/Tensorflow"

#################  4.分别在 host0/1 上建立容器 ps_host0/1:
echo building the containers of host$host
nvidia-docker run -it -v $SCRIPT_DIR:$WORKDIR -v $DATA_DIR:$DATA_DIR $limitresources $host_ip_cmd \
-p 5000:5000 -p 5001:5001 \
-e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" --name ps_host$host $image bash
