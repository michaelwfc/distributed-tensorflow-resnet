#!/bin/bash
ps_num=4
worker_num=4
for index in $(seq $ps_num)
do
	index=`expr $index - 1`
	docker stop tfps$index
	docker container rm tfps$index
done

for index in $(seq $worker_num)
do
	index=`expr $index - 1`
	docker stop tfworker$index
	docker container rm tfworker$index
	
done

docker stop tf-eval
docker container rm tf-eval
#docker stop tfworker4
docker container rm tfdocker


