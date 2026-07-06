
# General guidelines and recommendations for writing Dockerfile:
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/

################################################################################

# CUDA 12.1 + cuDNN 8 — matches torch==2.5.1 wheels from pytorch.org.
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Setup timezone.
RUN echo 'Asia/Tokyo' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Setup environment.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANGUAGE=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Python venv location — all pip commands inside the container use this.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# MuJoCo / OpenGL: use EGL for headless GPU rendering.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

################################################################################

# Install common essentials.
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    dirmngr \
    git \
    gnupg2 \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common optional tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    htop \
    inetutils-ping \
    iproute2 \
    less \
    mesa-utils \
    nano \
    net-tools \
    openssh-client \
    sshpass \
    usbutils \
    vim \
    && rm -rf /var/lib/apt/lists/*

################################################################################

# Install Python 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

################################################################################

# System libraries for OpenCV, OpenGL, video processing, and RealSense cameras.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa \
    libglew-dev \
    libosmesa6-dev \
    libsm6 \
    libusb-1.0-0-dev \
    libudev-dev \
    libxext6 \
    libxrender-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

################################################################################

# Create the Python virtual environment.
RUN python3.10 -m venv $VIRTUAL_ENV

# Install PyTorch with CUDA 12.1 first — xformers and other packages must see
# the correct torch version at install time.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

################################################################################

# Install Point-Policy Python dependencies.
WORKDIR /root/Point-Policy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# xformers must come after torch.
RUN pip install --no-cache-dir xformers==0.0.29.post1

################################################################################

# Copy submodule source trees so the editable installs below resolve correctly.
# At runtime the repo volume-mount overlays /root/Point-Policy/, so these
# editable links will transparently use the host-side code.
COPY co-tracker/   ./co-tracker/
COPY Franka-Teach/ ./Franka-Teach/
COPY franka-env/   ./franka-env/

RUN pip install --no-cache-dir -e "co-tracker/[all]" && \
    pip install --no-cache-dir -e Franka-Teach/ && \
    pip install --no-cache-dir -e franka-env/

# co-tracker extras installed separately in the original setup.sh.
RUN pip install --no-cache-dir flow_vis "imageio[ffmpeg]"

# Re-pin versions that downstream packages (robomimic, robosuite, etc.) tend to upgrade.
RUN pip install --no-cache-dir --force-reinstall \
    transformers==4.45.2 \
    huggingface_hub==0.25.2 \
    numpy==2.1.3

################################################################################

# Setup the Bash shell environment.

# Copy the helper scripts to the working directory.
COPY ./docker/dev/scripts/. /root/Point-Policy/docker/dev/scripts/
RUN chmod -R +x /root/Point-Policy/docker/dev/scripts/*

# Source the shell init script on every interactive session.
RUN echo "source /root/Point-Policy/docker/dev/scripts/initialize-bash-shell.sh" >> /root/.bashrc

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# Traceability: embed the Dockerfile's git commit hash in the image.
ARG DOCKERFILE_COMMIT_SHORT_SHA
ENV DOCKER_IMAGE_VERSION_POINT_POLICY=${DOCKERFILE_COMMIT_SHORT_SHA}

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
