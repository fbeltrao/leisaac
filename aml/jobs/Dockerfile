# docker build --tag isaac-lab-with-azcli:2.3.0 -f Dockerfile .
FROM nvcr.io/nvidia/isaac-lab:2.3.0

## Install UV from Astral SH
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash


# Install Docker CLI
RUN apt-get update && \
    apt-get install -y ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*



# Install CUDA development tools for building flash-attn
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-1 cuda-cudart-dev-12-1 libcublas-dev-12-1 libcusparse-dev-12-1 libcusolver-dev-12-1 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Install LeRobot for async inference
ENV LEROBOT_VENV_PATH="/root/lerobot" LEROBOT_PYTHON_PATH="/root/lerobot/bin/python"
# Dependencies for Groot: https://huggingface.co/docs/lerobot/en/groot
RUN uv venv --python 3.10 --no-project --no-config "$LEROBOT_VENV_PATH" && \
    uv pip install --python "$LEROBOT_PYTHON_PATH" "torch>=2.2.1,<2.8.0" "torchvision>=0.17.0,<0.21.0" --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --python "$LEROBOT_PYTHON_PATH" ninja "packaging>=24.2,<26.0" && \
    uv pip install --python "$LEROBOT_PYTHON_PATH" "setuptools>=80.9.0" && \
    uv pip install --python "$LEROBOT_PYTHON_PATH" "flash-attn>=2.5.9,<3.0.0" --no-build-isolation && \
    uv pip install --python "$LEROBOT_PYTHON_PATH" "lerobot[all,groot]==0.4.2"



# Accept EULA and Privacy Consent for Isaac Sim
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y

ENTRYPOINT ["/bin/bash"]

# resources:
    #   docker_args: -v=/var/run:/var/run # Needed for docker daemon to be available
    #   give all gpus: --gpus all