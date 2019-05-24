# NOTE: there is a bug with the latest torch version, resulting in a seg fault.
# Depending on which server you're running this on, you might need to modify the lines specifying cuda and pytorch versions.
# On some servers, that won't work either.
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

### --build-args
ARG UID=1000

### CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         sudo \
         vim \
         tmux \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         software-properties-common \
         && \
     rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7-dev

RUN curl --silent https://bootstrap.pypa.io/get-pip.py | python3.7
RUN pip3 install pipenv

#RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install python=3.6 && \
#     /opt/conda/bin/conda install numpy pyyaml scipy ipython cython mkl mkl-include && \
#     /opt/conda/bin/conda clean -ya
#ENV PATH /opt/conda/bin:$PATH

# Section to get permissions right, and avoid running inside as root {{
# Create a user matching the UID, and create/chmod home dir (== project directory)
# (uid corresponds to grehar in CS network)
# <<UID>> will be replaced by user id
RUN useradd -d /project -u $UID --create-home user && echo "user:pass" | chpasswd && adduser user sudo

RUN mkdir /project/app
ADD . /project/app

RUN chown -R user:user /project

USER user
WORKDIR /project/app

# Fix unicode for pipenv
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pipenv install && pipenv shell

#ENV PYTHONPATH "$PYTHONPATH:/project/"