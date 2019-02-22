#!/bin/bash
docker stop tfps0
docker stop tfworker0
docker stop tfworker1
docker stop tfworker2
docker stop tfworker3
docker stop tf-eval
#docker stop tfworker4
docker container rm tfdocker
docker container rm tfps0
docker container rm tfworker0
docker container rm tfworker1
docker container rm tfworker2
docker container rm tfworker3
docker container rm tfworker4
docker container rm tf-eval

#docker network rm tfdocker
for f in opt/*.csv; do tail -n +2 "$f" >> opt/temp.csv; done
echo "$(head -n 1 opt/0*.csv)
$(sort -t ',' -k 3 -g opt/temp.csv)" > ps1workers1.csv
rm -f opt/*.csv
