import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import datetime
import argparse
import sys
from scaler import *

sys.path.append('..')
from utils import visualization_utils as vis_util
from utils import label_map_util

RESULT_PATH = "/home/pi/Desktop/result/"
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

# Set up camera constants
IM_WIDTH = 640
IM_HEIGHT = 480

CAMERA_TYPE = 'picamera'

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected

#detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.

#detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

### Picamera ###

# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    # (boxes, scores, classes, num) = sess.run(
    #     [detection_boxes, detection_scores, detection_classes, num_detections],
    #     feed_dict={image_tensor: frame_expanded})
    num = sess.run(num_detections,
            feed_dict={image_tensor: frame_expanded})
    weigh = 0

    if num > 0:
        new_frame = cv2.resize(frame_expanded[0], (550, 250))
        weigh = scaler(np.expand_dims(new_frame, axis=0))[0][0]
        weigh = np.round(weigh, 2)
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        camera.capture(RESULT_PATH + "{}-{}.png".format(now, str(weigh)))

        print(weigh)

    # Draw the results of the detection (aka 'visulaize the results')

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.40)

    cv2.putText(frame,"scale: {}".format(weigh),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()
