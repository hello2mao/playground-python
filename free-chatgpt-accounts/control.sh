#!/bin/bash

IMAGE=chatgpt-accounts:latest
CONTAINER_NAME=chatgpt-accounts
mkdir -p /root/.cache/chatgpt-accounts

start() {
	sudo docker run -d \
        --restart=always --name ${CONTAINER_NAME} \
        -v /root/.cache/chatgpt-accounts:/app/data \
        ${IMAGE}
}

stop() {
	sudo docker rm ${CONTAINER_NAME} --force
}

case C"$1" in
    C)
        echo "Usage: $0 {start|stop|restart}"
        ;;
    Cstart)
        start
        echo "Start Done!"
        ;;
    Cstop)
        stop
        echo "Stop Done!"
        ;;
    Crestart)
        stop
        start
        echo "Restart Done!"
        ;;
    C*)
        echo "Usage: $0 {start|stop|restart}"
        ;;
esac