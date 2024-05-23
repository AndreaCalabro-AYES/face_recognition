# Use a multi-architecture base image
FROM python:3.10.3-slim

# Set the architecture-specific dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip

# Additional dependencies for Raspberry Pi (ARM)
RUN if [ "$(uname -m)" = "armv7l" ]; then \
    echo "Installing ARM-specific dependencies"; \
    apt-get update && apt-get install -y --fix-missing \
        libopenblas-dev; \
fi

# Additional dependencies for x86 (Windows/Mac)
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    echo "Installing x86_64-specific dependencies"; \
    apt-get update && apt-get install -y --fix-missing \
        libopenblas-dev; \
fi

# Install dlib
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Copy the face recognition app
COPY . /root/face_recognition
RUN cd /root/face_recognition && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

# Add pip3 install opencv-python==4.1.2.30 if you want to run the live webcam examples --> not working! Try with latest version
RUN pip3 install opencv-python==4.9.0.80

# Not needed as we use the docker-compose command
# CMD cd /root/face_recognition/examples && \
    # python3 recognize_faces_in_pictures.py