#!/usr/local/bin/zsh

mkdir tensorflow1 $$ cd tensorflow1

git clone --recurse-submodules https://github.com/tensorflow/models.git

echo export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow1/models/research/slim >> ~/.bashrc

cd ~/tensorflow1/models/research/object_detection

wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

cd ~/tensorflow1/models/research/

# it change every proto file to python (.py) file
protoc object_detection/protos/*.proto --python_out=.

cd ~/tensorflow1/models/research/object_detection

wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py



