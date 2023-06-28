# syntax = docker/dockerfile:1.3
FROM python:3.11
WORKDIR /app
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

COPY . .
CMD ["python3", "main.py"]