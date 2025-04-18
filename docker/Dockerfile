ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.11-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
ARG TORCHREC_VERSION=v0.7.0
ARG MEGATRON_REVISION=core_r0.9.0
ARG GITLAB_CLONE_ACCESS_TOKEN

WORKDIR /workspace/deps

RUN pip install --no-cache setuptools==69.5.1 setuptools-git-versioning scikit-build \
                typing-extensions iopath torchx black gin-config wandb torchmetrics==1.0.3
#                megatron-core==${MEGATRON_REVISION}
# Due to megatron-core doesn't have wheel package for python 3.12
# We can't install it from pypy, install from source as WAR
RUN git clone -b ${MEGATRON_REVISION} https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
    PY_ENV=pytorch:24.11 pip install -e ./megatron-lm
ENV PYTHONPATH="/workspace/deps/megatron-lm:$PYTHONPATH"

RUN git clone --recursive -b ${TORCHREC_VERSION} https://github.com/pytorch/FBGEMM.git fbgemm && \
    cd fbgemm/fbgemm_gpu && \
    python setup.py bdist_wheel --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0" && \
    python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0"
RUN rm -rf /workspace/deps/fbgemm

# import fbgemm_gpu requires libnvidia-ml
RUN test -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 || ln -s /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libnvidia-ml.so  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

RUN git clone --recursive -b pyhkv-0.1 https://${GITLAB_CLONE_ACCESS_TOKEN}@gitlab-master.nvidia.com/aleliu/torchrec.git torchrec && \
    cd torchrec && \
    pip install --no-deps . && \
    cd ../ && rm -rf torchrec

RUN git clone --recursive -b main https://${GITLAB_CLONE_ACCESS_TOKEN}@gitlab-master.nvidia.com/aleliu/hstu.git hstu && \
    cd hstu && \
    git checkout 9a90a45db68f99c769baa3e705863dc410cc737e && \
    HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=TRUE HSTU_DISABLE_DRAB=TRUE pip install . && \
    cd hopper && \
    HSTU_DISABLE_SM8x=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=TRUE HSTU_DISABLE_DRAB=TRUE pip install . && \
    cd ../ && rm -rf hstu

RUN git clone --recursive -b main https://${GITLAB_CLONE_ACCESS_TOKEN}@gitlab-master.nvidia.com/Devtech-Compute/dynamicemb.git dynamicemb && \
    cd dynamicemb && \
    python setup.py install && \
    cd ../

RUN test -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 && rm -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# For dev
RUN apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache pre-commit

FROM ${DEVEL_IMAGE} AS build
WORKDIR /workspace/distributed-recommender
COPY . .

RUN ls && python3 setup.py install

