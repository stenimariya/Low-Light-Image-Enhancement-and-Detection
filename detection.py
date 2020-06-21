import cv2
import argparse
import sys
import numpy as np
import os.path

weight_path = "{base_path}/yolov3.weights".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
# weight_path="my_weights/yolov3_4000.weights"Z

detection_no = []
# Initialize the parameters
confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.2  # Non-maximum suppression threshold

inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image
# Load names of classes
print("111")
modelConfiguration = "{base_path}/yolov3.cfg".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

classesFile = "{base_path}/classes.names".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.



modelWeights = weight_path;

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    #    cv2.imshow('caaon',img[top:bottom,right:left])
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #    print(top,bottom,right,left)
    cv2.circle(img, (left, top), 2, (255, 0, 0), 2)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # Display the label at the top of the bounding box

    # Remove the bounding boxes with low confidence using non-maxima suppression


def postprocess(frame, outs):
    copy_frame = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        #        print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            #            if detection[4]>confThreshold:
            #                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)

            if confidence > confThreshold and classes[classId] == 'car' or classes[classId] == 'person' or classes[
                classId] == 'truck' or classes[classId] == 'traffic light':
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        detection_no.append([classIds[i], confidences[i], left, top, left + width, top + height])
        drawPred(copy_frame, classIds[i], confidences[i], left, top, left + width, top + height)
    return copy_frame


def detect(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    final_image = postprocess(img, outs)
    return final_image

# print(len(detection_no))
# # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
# t, _ = net.getPerfProfile()
# label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
# img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
# cv2.imshow('image', img)
# cv2.waitKey(0)
