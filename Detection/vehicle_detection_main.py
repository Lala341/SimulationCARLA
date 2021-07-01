
# Imports
from typing import SupportsComplex
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import zipfile
import cv2
import numpy as np
import csv
import time
from packaging import version

from collections import defaultdict
from io import StringIO
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# initialize .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'id,Vehicle Type/Size,Vehicle Color'
    writer.writerows([csv_line.split(',')])

with open('traffic_measurementlocations.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'id,time,x,y'
    writer.writerows([csv_line.split(',')])
# input video
source_video = 'test.avi'
cap = cv2.VideoCapture(source_video)


# Variables
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("width")
print(width)
print("height")
print(height)
  
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 93

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #od_graph_def = tf.compat.v1.GraphDef() # use this line to run it with TensorFlow version 2.x
    #with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: # use this line to run it with TensorFlow version 2.x
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)
def convertCoordinates(top,bottom,right,left):
    px=((right-left)/2)+left
    py=((bottom-top)/2)+top
    return [px,py]
def saveObjects(objects):
    with open('traffic_measurement.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(0,len(objects)):
            writer.writerows([[i,objects[i]["type"], objects[i]["color"] ]])
    with open('traffic_measurementlocations.csv', 'a') as l:
        writer = csv.writer(l)
        for i in range(0,len(objects)):
            objectcurrent=objects[i]
            timec=int(objectcurrent["locations"][0][1])
            sumx=0
            sumy=0
            countc=0
            for j in objectcurrent["locations"]:
                if(timec==int(j[1])):
                    temp=convertCoordinates(j[2],j[3],j[4],j[5])
                    sumx+=temp[0]
                    sumy+=temp[1]
                    countc+=1
                else:
                    writer.writerows([[j[0],timec,sumx/countc,sumy/countc]])
                    timec=int(j[1])
                    temp=convertCoordinates(j[2],j[3],j[4],j[5])
                    sumx=temp[0]
                    sumy=temp[1]
                    countc=1
            writer.writerows([[j[0],timec,sumx/countc,sumy/countc]])
            

def sameObject(csv_line, objects,timeo):
    (size, color, direction, speed, top,bottom,right,left) = \
                            csv_line.split(',')
    left=float(left)
    right=float(right)
    top=float(top)
    bottom=float(bottom)
    add=False
    for i in range(0,len(objects)):
        if(add==False and objects[i]["type"]==size):
            ll = objects[i]["locations"][-1]
            if(timeo-ll[1]<1.1):
                areatotal=(top-bottom)*(right-left)
                areacruzada=(max(top,ll[2])-min(bottom, ll[3]))*(min(right,ll[4])-max(left, ll[5]))
                if((areacruzada/areatotal)>0.50):
                    objects[i]["locations"].append([i,timeo,top,bottom,right,left])
                    add=True
    if(add==False):
        objects.append({
            "type":size,
            "color":color,
            "locations":[[len(objects),timeo,top,bottom,right,left]]
        })
# Detection
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))
def object_detection_function(command):
    total_passed_vehicle = 0

    if(command=="imwrite"):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(source_video.split(".")[0]+'_output.avi', fourcc, fps, (width, height))
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
        #with tf.compat.v1.Session(graph=detection_graph) as sess: # use this line to run it with TensorFlow version 2.x

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            objects=[]
            print("fps")
            print(fps)
            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()
                cframe = cap.get(cv2.CAP_PROP_POS_FRAMES) 
                if not ret:
                    print ('end of the video file...')
                    break
                size=frame.shape
                input_frame = frame
                time = (cframe / 1000)*60

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                (counter, csv_lines) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    )

                total_passed_vehicle = total_passed_vehicle + len(csv_lines)

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Time (seg):  ' + str(time),
                    (10, 55),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                
                # insert information text to video frame
                

                if(command=="imshow"):
                    cv2.imshow('vehicle detection', input_frame)
                    cv2.setMouseCallback('vehicle detection', onMouse)
                    

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                elif(command=="imwrite"):
                    output_movie.write(input_frame)
                    print("writing frame...")

                if len(csv_lines) >0:
                    for csv_line in csv_lines:
                        sameObject(csv_line, objects,time)
                if(time>305):
                    cap.release()
                    cv2.destroyAllWindows()
                    saveObjects(objects)
            

    
            


import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(description='Vehicle Detection TensorFlow.')
parser.add_argument("command",
                    metavar="<command>",
                    help="'imshow' or 'imwrite'")
args = parser.parse_args()
object_detection_function(args.command)		
