#!/bin/bash

sudo apt-get update #update

sudp apt-get -y dist-upgrade # upgrade

# install tensorflow 
mkdir ~/tf && cd ~/tf

wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.13.1/tensorflow-1.13.1-cp35-none-linux_armv7l.whl

sudo pip3 install tensorflow

# tensorflow needs libatlas
sudo apt-get -y install libatlas-base-dev

# install python packages 
sudo pip3 install matplotlib pillow jupyter cython

# get dependency for lxml
sudo apt-get install -y libxml2-dev libxslt-dev
sudo pip3 install lxml

# just checking
sudo apt-get install python-tk

##OpenCV

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev

sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

sudo apt-get -y install libxvidcore-dev libx264-dev

sudo apt-get -y install qt4-dev-tools

sudo pip3 install opencv-python 

# ProtoBuf 아래 링크 참조 
# http://osdevlab.blogspot.com/2016/03/how-to-install-google-protocol-buffers.html

sudo apt-get -y install autoconf automake libtool curl

wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protobuf-all-3.8.0.tar.gz

tar -zxvf protobuf-all-3.8.0.tar.gz

cd protobuf-3.8.0 && ./configure

make  # take long time about over 60 minutes 

make check # even longer...

sudo make install 

cd python 
export LD_LIBRARY_PATH=../src/.libs 
python3 setup.py build --cpp_implementation
python3 setup.py test --cpp_implementation

sudo python3 setup.py install --cpp_implementation

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3

sudo ldconfig

sudo reboot now
######## After that set tensorflow directory
