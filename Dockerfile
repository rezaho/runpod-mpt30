# FROM ubuntu:22.04
# FROM --platform=linux/amd64 python:3.10-slim-bullseye
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY . /app
WORKDIR /app

RUN apt-get update
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get install build-essential -y
RUN apt-get install cmake -y
# RUN apt-get install llvm -y
# RUN apt-get install libsasl2-dev python3-dev libldap2-dev libssl-dev -y
# RUN apt-get install python3.11 -y
RUN apt-get install python3-pip -y

RUN apt-get install git -y

RUN pip install --upgrade pip
RUN pip install runpod packaging wheel cmake
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
# RUN git clone https://github.com/HazyResearch/flash-attention.git --branch v1.0.7
# RUN python3 ./flash-attention/setup.py bdist_wheel
RUN pip install flash_attn-1.0.7-cp310-cp310-linux_x86_64.whl
RUN pip install triton_pre_mlir-2.0.0-cp310-cp310-linux_x86_64.whl
RUN pip install numpy scipy transformers peft loralib bitsandbytes datasets accelerate einops triton


# # pip install before installing the llm-foundry
# RUN pip install packaging
# RUN pip install torch
# # packaging, torch, 
# # Install and uninstall foundry to cache foundry requirements
# RUN git clone -b main https://github.com/mosaicml/llm-foundry.git && \
#     pip install --no-cache-dir "./llm-foundry[gpu]" && \
#     pip uninstall -y llm-foundry && \
#     rm -rf llm-foundry

# RUN pip install runpod
# Install production dependencies.
# ADD requirements.txt .
RUN pip install -r ./requirements.txt

# ADD handler.py .


CMD [ "python3", "-u", "./handler.py" ]