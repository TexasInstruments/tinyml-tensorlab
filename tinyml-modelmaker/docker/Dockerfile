# base image
ARG DOCKER_REPO_LOCATION=""
FROM ${DOCKER_REPO_LOCATION}ubuntu:22.04

# user
# we are not setting a user using the USER command, instead docker run would use the user id of the calling user
# in that case, HOME enviroment variable will not be set and we need to set it explicity here.
ENV USER_NAME=tinyml
ENV HOME=/opt/tinyml
ENV HOME_DIR=${HOME}

ENV TOOLS_PATH=${HOME_DIR}/bin
ENV INSTALLER_PATH=${HOME_DIR}/installer
RUN mkdir -p ${HOME_DIR} && \
    mkdir -p ${TOOLS_PATH} && \
    mkdir -p ${INSTALLER_PATH} && \
    echo "PWD=$(pwd)"

# build settings
ARG PROJECT_NAME="tinyml-modelmaker"
ARG DEBIAN_FRONTEND=noninteractive
# avoid warnings/errors while installing pillow-simd
ARG LC_ALL=C.UTF-8
# proxy
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy=""

# proxy for apt
# /etc/apt/apt.conf will have to be updated if apt install is needed during docker run
# and if the proxy used is different from the proxy being used here during docker build
RUN if [ ! -z $http_proxy ]; then echo "Acquire::http::proxy \"${http_proxy}\";" > /etc/apt/apt.conf; fi && \
    if [ ! -z $https_proxy ]; then echo "Acquire::https::proxy \"${https_proxy}\";" >> /etc/apt/apt.conf; fi

# baseline
RUN apt update && \
    apt install -y sudo g++ gcc git curl iputils-ping wget cmake build-essential libgtk-3-0 apt-utils libgl1 llvm llvm-dev \
    libbz2-dev xz-utils libreadline-dev libncurses5-dev libssl-dev libsqlite3-dev liblzma-dev \
    libffi-dev libjpeg-dev zlib1g-dev graphviz graphviz-dev protobuf-compiler

# ##################################################################
# change working directory
WORKDIR ${HOME_DIR}

# pyenv install
ENV PYENV_ROOT ${HOME_DIR}/.pyenv
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# we define PIP_CACHE_DIR here to point to a writable location
# to enable container user to be different by using the docker run option --user
RUN echo ' ' >> ${HOME_DIR}/.pyenvrc && \
    echo "export PIP_CACHE_DIR=${HOME_DIR}/.cache/pip" >> ${HOME_DIR}/.pyenvrc && mkdir -p ${HOME_DIR}/.cache/pip

# set paths and activate env
# .bashrc is not sourced in non interactive mode - so write it to another file
RUN echo ' ' >> ${HOME_DIR}/.pyenvrc && \
    echo '# pyenv settings' >> ${HOME_DIR}/.pyenvrc && \
    echo "command -v pyenv >/dev/null || export PATH=:$PYENV_ROOT/bin:$PATH" >> ${HOME_DIR}/.pyenvrc && \
    echo 'eval "$(pyenv init -)"' >> ${HOME_DIR}/.pyenvrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME_DIR}/.pyenvrc && \
    echo ' ' >> ${HOME_DIR}/.pyenvrc

# .profile is needed for non-interactive shell spawns
RUN echo ' ' >> ${HOME_DIR}/.profile && \
    echo "command -v pyenv >/dev/null || source ${HOME_DIR}/.pyenvrc" >> ${HOME_DIR}/.profile && \
    echo ' ' >> ${HOME_DIR}/.profile

# .bashrc is needed for interactive shell spawns
RUN echo ' ' >> ${HOME_DIR}/.bashrc && \
    echo "command -v pyenv >/dev/null || source ${HOME_DIR}/.pyenvrc" >> ${HOME_DIR}/.bashrc && \
    echo ' ' >> ${HOME_DIR}/.bashrc

# pyenv works with bash, so change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# create virtualenv
RUN source ${HOME_DIR}/.bashrc && \
    pyenv install 3.10 && \
    pyenv virtualenv 3.10 py310 && \
    pyenv rehash && \
    echo 'pyenv activate py310' >> ${HOME_DIR}/.pyenvrc && \
    echo ' ' >> ${HOME_DIR}/.pyenvrc

# update pip setuptools
RUN source ${HOME_DIR}/.bashrc && \
    python -m ensurepip --upgrade && \
    python -m pip install --no-input --upgrade pip setuptools && \
    python -m pip install --no-input --upgrade wheel cython numpy

#################################################################################
# Note 1: this can be reversed later as a last step after all installations are done.
# enabling write permissions for all users, if support is needed for users passed via --user argument of docker run
# RUN chmod -R ugo+rwx ${HOME_DIR}

# For Path variables
# Note 3: in the container, .bashrc or .pyenvrc may need to be sourced to activate the Python environment
# if it is not being activated automatically.
RUN source ${HOME_DIR}/.bashrc
