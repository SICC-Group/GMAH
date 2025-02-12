# 使用 pytorch 基础镜像（Ubuntu 20.04）
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# FROM ${BASE_IMAGE} AS downloader
FROM ${BASE_IMAGE}

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

# Install Webots runtime dependencies
RUN apt-get update && apt-get install --yes \
 vim \
 locales \
 python3-pip \
 python3-dev \
 python3-venv \
 python3-setuptools 

# install NVIDIA drivers and CUDA
# RUN mkdir -p /etc/systemd/system && \
#  echo "[Service]\nEnvironment=PATH=/usr/local/cuda/bin:$PATH\n" > /etc/systemd/system/docker.service.d/override.conf

RUN apt-get update && apt-get install -y \
 nvidia-cuda-toolkit && \
 rm -rf /var/lib/apt/lists/*

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES graphics,compute,utility

# Set a user name to fix a warning
ENV USER root
WORKDIR /root

COPY requirements.txt .
RUN pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --no-cache-dir -r requirements.txt

# 复制整个项目文件到容器，包括 custom_package
COPY . .

# 在readme里写pip install -e.
#RUN pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple /root


# Set the locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

EXPOSE 22

# Finally open a bash command to let the user interact
CMD ["/bin/bash"]