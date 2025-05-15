ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.7.1
ARG CUDA_ARCHITECTURES=75;86
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libcurl4-openssl-dev \
    python3 python3-pip python3-dev python3-venv \
    cuda-cudart-dev-11-7 cuda-libraries-dev-11-7 cuda-nvcc-11-7 cuda-compiler-11-7 \
    ffmpeg lsof wget sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch and HLoc dependencies in virtual environment
RUN pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir kornia==0.6.11 h5py py360convert

# Clone and install HLoc from tag 1.5
RUN git clone https://github.com/cvg/Hierarchical-Localization.git /hloc && \
    cd /hloc && \
    git checkout 1.5 && \
    sed -i '/lightglue/d' requirements.txt && \
    pip install --no-cache-dir . && \
    rm -rf /hloc

# Build and install COLMAP
RUN git clone https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    git checkout main && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/colmap-install -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja install && \
    cd ../.. && rm -rf /colmap

# Runtime stage
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-program-options1.74.0 libboost-filesystem1.74.0 libboost-graph1.74.0 libboost-system1.74.0 \
    libc6 libceres2 libfreeimage3 libgcc-s1 libgflags2.2 \
    libgl1 libglew2.2 libgoogle-glog0v5 libqt5core5a libqt5gui5 libqt5widgets5 \
    libcurl4 python3 python3-pip python3-venv xvfb libx11-6 libxext6 libxrender1 x11-utils \
    cuda-cudart-11-7 cuda-libraries-11-7 ffmpeg lsof sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in virtual environment
RUN pip install --no-cache-dir flask psutil gunicorn GPUtil plyfile pycolmap numpy==1.26.4 scipy opencv-contrib-python \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir kornia==0.6.11 h5py py360convert

# Install HLoc from tag 1.5 in runtime stage
RUN git clone https://github.com/cvg/Hierarchical-Localization.git /hloc && \
    cd /hloc && \
    git checkout 1.5 && \
    sed -i '/lightglue/d' requirements.txt && \
    pip install --no-cache-dir . && \
    rm -rf /hloc

# Copy COLMAP installation
COPY --from=builder /colmap-install/ /usr/local/

# Copy application code
WORKDIR /app
COPY app.py .
COPY static/ static/
COPY vocab_tree.bin /app/vocab_tree.bin
COPY static/alarm.mp3 /app/static/alarm.mp3
RUN chmod 644 /app/vocab_tree.bin

# Verify static files
RUN ls -l /app/static/ && test -f /app/static/index.js && test -f /app/static/index.html

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PYTHONPATH=/app:/

# Expose port
EXPOSE 8080

# Run Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "36000", "--limit-request-line", "8192", "--limit-request-fields", "2000", "app:app"]