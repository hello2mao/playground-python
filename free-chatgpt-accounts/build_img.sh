#!/bin/sh
IMAGE=chatgpt-accounts:latest
export DOCKER_BUILDKIT=1
docker build -t "${IMAGE}" .
