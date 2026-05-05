FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# minerl v1.0 requires Java 8 specifically
RUN apt-get update && apt-get install -y \
    software-properties-common curl git ffmpeg \
    openjdk-8-jdk \
    libgl1-mesa-glx libosmesa6 libglfw3 \
    xvfb x11-utils \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

WORKDIR /workspace

COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install -r requirements.txt && \
    python3.10 -m pip install git+https://github.com/minerllabs/minerl

COPY . .

ENV DISPLAY=:1

ENTRYPOINT ["/workspace/run_minerl.sh"]
CMD ["test_minerl.py"]
