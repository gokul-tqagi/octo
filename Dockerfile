# Dockerfile: Full Octo container for data extraction (mcap→RLDS) and finetuning.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# cuPTI for JAX CUDA backend
ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

WORKDIR /octo

# Install octo dependencies (installs jax 0.4.20 CPU first)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Override with CUDA-enabled jaxlib matching jax 0.4.20
RUN pip3 install --no-cache-dir "jaxlib==0.4.20+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Pin transformers to version with Flax support (5.x dropped FlaxAutoModel)
RUN pip3 install --no-cache-dir "transformers==4.36.2"

# Data extraction deps: mcap-ros2-support for bag parsing, opencv for image processing
RUN pip3 install --no-cache-dir \
    mcap mcap-ros2-support \
    opencv-python-headless \
    scipy==1.11.4 \
    open3d

# Install octo package
COPY . .
RUN pip3 install --no-cache-dir -e .

CMD ["bash"]
