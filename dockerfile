ARG UBUNTU_VERSION=22.04

# Builder stage
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libcurl4-openssl-dev \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Build SphereSfM (integrated with COLMAP)
RUN git clone https://github.com/json87/SphereSfM.git colmap && \
    cd colmap && \
    git checkout main && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/colmap-install && \
    ninja install && \
    rm -rf /colmap

# Runtime stage
FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-program-options1.74.0 libboost-filesystem1.74.0 libc6 libceres2 libfreeimage3 libgcc-s1 \
    libgl1 libglew2.2 libgoogle-glog0v5 libqt5core5a libqt5gui5 libqt5widgets5 \
    libcurl4 python3 python3-pip xvfb libx11-6 libxext6 libxrender1 x11-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install flask

# Copy SphereSfM/COLMAP installation
COPY --from=builder /colmap-install/ /usr/local/

# Copy application code
WORKDIR /app
COPY app.py .
COPY static/ static/

# Verify static files
RUN ls -l /app/static/ && test -f /app/static/index.js && test -f /app/static/index.html

# Expose port
EXPOSE 8080

# Run Flask development server
CMD ["python3", "app.py"]