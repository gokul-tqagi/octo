# Dockerfile: Dependencies-only base image for Octo finetuning and data extraction.
# Project code and data are MOUNTED at runtime, never baked in.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Octo Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# CUDA-enabled jaxlib
RUN pip3 install --no-cache-dir "jaxlib==0.4.20+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Pin transformers for Flax support
RUN pip3 install --no-cache-dir "transformers==4.36.2"

# Data extraction + validation + config deps
RUN pip3 install --no-cache-dir \
    mcap mcap-ros2-support \
    opencv-python-headless \
    scipy==1.11.4 \
    open3d plotly pyarrow pyyaml

# Project code mounted at /project at runtime.
# Entrypoint runs pip install -e to register the package before the command.
WORKDIR /octo

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
