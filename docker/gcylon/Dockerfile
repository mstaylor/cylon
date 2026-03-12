# gcylon Dockerfile - GPU-accelerated Cylon with cuDF/RMM support
# Uses NVIDIA CUDA base image + conda packages + source builds
# RTX 5090 (Blackwell sm_120) requires CUDA 12.8+
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
LABEL maintainer="cylondata@googlegroups.com"

ARG DEBIAN_FRONTEND=noninteractive
ARG CYLON_BRANCH=cylon-rust-continued

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

ENV CYLON_HOME=/cylon
ENV CYLON_BUILD=/cylon/cpp/build_cylon
ENV GCYLON_BUILD=/cylon/cpp/build_gcylon
ENV CONDA_PREFIX=/opt/conda/envs/cylon_dev
ENV UCX_HOME=/ucx
ENV UCC_HOME=/ucc

ADD https://google.com cache_bust

WORKDIR $CYLON_HOME

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate cylon_dev" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=$CYLON_PREFIX"/lib >> ~/.bashrc && \
    /opt/conda/bin/conda install -n base conda-libmamba-solver -y && \
    /opt/conda/bin/conda config --set solver libmamba



RUN git clone https://github.com/mstaylor/cylon.git $CYLON_HOME
RUN cd $CYLON_HOME && git checkout -b ${CYLON_BRANCH} origin/${CYLON_BRANCH}



SHELL ["/bin/bash", "--login", "-c"]

# Use the working cylon_NoUCX.yml environment (avoids conda channel conflicts)
RUN conda env create -f $CYLON_HOME/conda/environments/cylon_NoUCX.yml

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    conda install -c anaconda boto3 -y && \
    conda install -c conda-forge redis-py -y && \
    pip install cloudmesh-openstack

# Install cuDF/RMM from rapidsai (CUDA 12.8 for RTX 5090 Blackwell support)
# Pin to RAPIDS 24.10 for consistent CCCL compatibility
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    conda install -c rapidsai -c conda-forge -c nvidia \
        cudf=24.10 libcudf=24.10 rmm=24.10 librmm=24.10 \
        cuda-version=12.8 -y && \
    echo "=== Installed versions ===" && \
    conda list | grep -E "cudf|rmm|cccl"

# Remove conda UCC/UCX libraries to avoid conflicts with custom builds
RUN rm -f /opt/conda/envs/cylon_dev/lib/libucc* /opt/conda/envs/cylon_dev/lib/libucx* \
         /opt/conda/envs/cylon_dev/lib/libucp* /opt/conda/envs/cylon_dev/lib/libucs* \
         /opt/conda/envs/cylon_dev/lib/libuct*

# Ensure numpy is installed using same channels as cuDF for compatibility
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    conda install -c rapidsai -c conda-forge -c nvidia numpy cython setuptools -y && \
    python -c "import numpy; print(f'numpy {numpy.__version__} installed')"

RUN apt-get update &&  apt-get install autoconf -y &&  apt-get install libtool -y && apt-get install cmake -y && \
    apt-get install libnuma-dev -y

# Build UCX from source (includes support for overriding device eth address)
RUN git clone --single-branch -b v1.19.1 https://github.com/openucx/ucx.git $UCX_HOME && \
    cd $UCX_HOME && \
    ./autogen.sh && \
    ./configure --prefix=$UCX_HOME/install && \
    make -j$(nproc) install

# Build UCC from source
RUN git clone --single-branch -b v1.6.0 https://github.com/openucx/ucc.git $UCC_HOME && \
    cd $UCC_HOME && \
    ./autogen.sh && \
    ./configure --prefix=$UCC_HOME/install --with-ucx=$UCX_HOME/install && \
    make -j$(nproc) install

# Build Redis (hiredis + redis-plus-plus)
ENV HIREDIS_HOME=/hiredis
ENV REDIS_PLUS_PLUS_HOME=/redis-plus-plus

#REDIS install (override LIBRARY_PATH for hiredis to prevent NVIDIA path corruption)
RUN git clone https://github.com/redis/hiredis.git /hiredis && \
    cd /hiredis && \
    make && \
    make PREFIX=/usr/local LIBRARY_PATH=lib install && \
    ldconfig && \
    git clone https://github.com/sewenew/redis-plus-plus.git /redis-plus-plus && \
    cd /redis-plus-plus && \
    mkdir build && \
    cd build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make && make install

# Build cylon (CPU library) with UCX, UCC, and Redis support using build.py
ENV CYLON_PREFIX=/cylon/install
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd $CYLON_HOME && \
    python build.py -cmake-flags="-DCYLON_UCX=1 -DUCX_INSTALL_PREFIX=$UCX_HOME/install -DCYLON_UCC=1 -DUCC_INSTALL_PREFIX=$UCC_HOME/install -DCYLON_USE_REDIS=1" -ipath="$CYLON_PREFIX" --cpp --python

# Build and install gcylon (GPU library) to conda prefix
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    export CYLON_LIB_DIR=${CYLON_PREFIX}/lib && \
    mkdir -p ${GCYLON_BUILD} && \
    cd ${GCYLON_BUILD} && \
    cmake ${CYLON_HOME}/cpp/src/gcylon \
        -DCYLON_VERSION=0.7.0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
        -DCMAKE_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
        -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
        -DGCYLON_BUILD_TESTS=ON && \
    make -j$(nproc) && \
    make install

# Set library paths (needed for pygcylon build)
ENV LD_LIBRARY_PATH="${GCYLON_BUILD}/lib:${CYLON_PREFIX}/lib:${CONDA_PREFIX}/lib:${UCX_HOME}/install/lib:${UCC_HOME}/install/lib:${HIREDIS_HOME}/install/lib:${REDIS_PLUS_PLUS_HOME}/install/lib"
ENV GCYLON_LIB_DIR="${GCYLON_BUILD}/lib"
ENV CYLON_LIB_DIR="${CYLON_PREFIX}/lib"

# Build pygcylon (Python bindings for gcylon GPU library)
# Note: CYLON_PREFIX/lib needed for libcylon, gcylon is now in CONDA_PREFIX/lib
ENV PYGCYLON_SOURCE_DIR=${CYLON_HOME}/python/pygcylon
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    echo "=== Checking CCCL installation ===" && \
    conda list | grep -i cccl && \
    echo "=== Looking for cuda::mr headers ===" && \
    find $CONDA_PREFIX/include -name "memory_resource*" -type f 2>/dev/null | head -10 && \
    find $CONDA_PREFIX/include -path "*cuda*mr*" -type f 2>/dev/null | head -10 && \
    ls -la $CONDA_PREFIX/include/cuda/ 2>/dev/null || echo "No cuda/ dir" && \
    echo "=== End CCCL check ===" && \
    cd ${PYGCYLON_SOURCE_DIR} && \
    export CUDA_HOME=/usr/local/cuda && \
    export LIBRARY_PATH="${CYLON_PREFIX}/lib:${LIBRARY_PATH}" && \
    python setup.py build_ext --inplace && \
    pip install --no-build-isolation .

# Add LD_LIBRARY_PATH to bashrc for interactive shells
RUN echo "export LD_LIBRARY_PATH=${GCYLON_BUILD}/lib:${CYLON_PREFIX}/lib:\${CONDA_PREFIX}/lib:${UCX_HOME}/install/lib:${UCC_HOME}/install/lib:${HIREDIS_HOME}/install/lib:${REDIS_PLUS_PLUS_HOME}/install/lib" >> ~/.bashrc

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build Rust crate with GPU feature
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd ${CYLON_HOME}/rust && \
    cargo build --features gpu --examples --release

# Default command - run bash with conda activated
CMD ["/bin/bash", "--login"]
