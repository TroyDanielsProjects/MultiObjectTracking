
"""
Author: Troy Daniels

notes:
Images should be in a list where each image is of type numpy.ndarray

"""
import cv2 as cv
import numpy as np
import math
import random
import tensorflow as tf
import copy
from tensorflow import keras

def load_data():
    # lists for data and labels
    images = []
    bboxes = []
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    print("Starting to load data")
    for line in ofile:
        bboxdata =line.split(",")
        x1 = int(bboxdata[1])
        y1 = int(bboxdata[2])
        x2 = int(bboxdata[3])
        y2 = int(bboxdata[4])
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        # get the size of the image. Resize the image. Make the bounding box numbers => {0,1}
        H, W, C = image.shape
        image = cv.resize(image, (256,256))
        image = (image - 127.5)/127.5
        images.append(image)
        norm_x1 = x1 / W
        norm_y1 = y1 / H
        norm_x2 = x2 / W
        norm_y2 = y2 / H
        # add the data/label to the lists
        bbox = [norm_x1,norm_y1,norm_x2,norm_y2]
        bboxes.append(bbox)
    # create the training ,testing and validation
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(bboxes[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(bboxes[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(bboxes[round(len(images)*0.9):])
    print("Finished loading data")
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)


# - objectiveness score - goes through each image and segments it out into many while computing the lable , how
# much of the object is in the segement which is computed using the ground truth bounding box
def splitImages(filter_zero_lables=True, skip_data=0):
     # lists for data and labels
    images = []
    labels = []
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox_small.csv")
    ofile.readline()
    print("Starting to load data")
    if skip_data>0:
        for i in range(skip_data):
            ofile.readline()
    for line in ofile:
        bboxdata =line.split(",")
        x1_bbox = int(bboxdata[1])
        y1_bbox = int(bboxdata[2])
        x2_bbox = int(bboxdata[3])
        y2_bbox = int(bboxdata[4])
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        H, W, C = image.shape
        one_sixth_image_height =  math.floor(H/6)
        one_sixth_image_width =  math.floor(W/6)
        segment_dim_height =  math.floor(H/2)
        segment_dim_width = math.floor(W/2)
        for i in range(segment_dim_height, H ,one_sixth_image_height):
            for j in range(segment_dim_width, W, one_sixth_image_width  ) :
                temp_image = image[i-segment_dim_height:i,j-segment_dim_width:j]
                #coordinates of segment
                x1 = j - segment_dim_width
                y1 = i - segment_dim_height
                x2 = j
                y2 = i
                # coordinates of intersection of segment and bonding box (may be no intersection which is checked below)
                x1_intersect = max(x1,x1_bbox)
                y1_intersect = max(y1,y1_bbox)
                x2_intersect = min(x2, x2_bbox)
                y2_intersect = min(y2, y2_bbox)
                
                if (x1_intersect>x2) or (y1_intersect>y2) or (x2_intersect<x1) or (y2_intersect<y1):
                    objectiveness_label = 0
                else:
                    objectiveness_label = ( (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect) ) / ( (x2_bbox-x1_bbox) * (y2_bbox - y1_bbox) )

                temp_image = cv.resize(temp_image, (256,256))
                temp_image = (temp_image - 127.5) / 127.5
                show_segment = copy.deepcopy(temp_image)
                images.append(temp_image)
                labels.append(objectiveness_label)
    # The data is biased toward some lables depending on the size of the segment, this gets rid of the bias in the data
    # if filter_zero_lables:
    #     images,labels = filter_data(images,labels)
    # Simular data are also consecutive in the data, this shuffles it
    # images, labels = shuffleData(images,labels)
    # break into train segment and test
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(labels[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(labels[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(labels[round(len(images)*0.9):])
    print("Finished loading data")
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)

# - objectiveness score
def filter_data(images,labels):
    distribution = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(labels)):
        amount = round(labels[i]*10)
        distribution[amount]+=1
    print(distribution)
    distributionSorted = copy.deepcopy(distribution)
    distributionSorted.sort()
    medianAmount = distributionSorted[5]
    labelsToReturn = []
    imagesToReturn = []
    for i in range(len(images)):
        amount = round(labels[i]*10)
        if distribution[amount] > medianAmount:
            distribution[amount]-=1
        else:
            labelsToReturn.append(labels[i])
            imagesToReturn.append(images[i])
    return imagesToReturn,labelsToReturn

        
def IoU(bbox_true, bbox_pred):

    x1_true = bbox_true[0]
    y1_true = bbox_true[1]
    x2_true = bbox_true[2]
    y2_true = bbox_true[3]

    x1_pred = bbox_pred[0]
    y1_pred = bbox_pred[1]
    x2_pred = bbox_pred[2]
    y2_pred = bbox_pred[3]


    x1_intersect = max(x1_true,x1_pred)
    y1_intersect = max(y1_true,y1_pred)
    x2_intersect = min(x2_true,x2_pred)
    y2_intersect = min(y2_true,y2_pred)


    if (x1_intersect>x2_true) or (y1_intersect>y2_true) or (x2_intersect<x1_true) or (y2_intersect<y1_true):
        intersection = 0
    else:
        intersection = ( (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect) )
    union = ( (x2_true - x1_true) * (y2_true - y1_true) + (x2_pred - x1_pred) * (y2_pred - y1_pred) - intersection)
    return intersection / union


def normalizedOverlap(bbox_true, bbox_pred):

    x1_true = bbox_true[0]
    y1_true = bbox_true[1]
    x2_true = bbox_true[2]
    y2_true = bbox_true[3]

    x1_pred = bbox_pred[0]
    y1_pred = bbox_pred[1]
    x2_pred = bbox_pred[2]
    y2_pred = bbox_pred[3]


    x1_intersect = max(x1_true,x1_pred)
    y1_intersect = max(y1_true,y1_pred)
    x2_intersect = min(x2_true,x2_pred)
    y2_intersect = min(y2_true,y2_pred)


    if (x1_intersect>x2_true) or (y1_intersect>y2_true) or (x2_intersect<x1_true) or (y2_intersect<y1_true):
        intersection = 0
    else:
        intersection = ( (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect))
    normalizingSize = min ( (x2_true - x1_true) * (y2_true - y1_true), (x2_pred - x1_pred) * (y2_pred - y1_pred))
    return intersection / normalizingSize


# - objectiveness score
def shuffleData(list_data,list_labels):
    listOfIndexes = []
    dataToReturn = []
    labelsToReturn = []
    for i in range(len(list_data)):
        listOfIndexes.append(i)
    for i in range(len(list_data)):
        randIndex = random.randint(0,len(listOfIndexes)-1)
        randNum = listOfIndexes.pop(randIndex)
        dataToReturn.append(list_data[randNum])
        labelsToReturn.append(list_labels[randNum])
    return dataToReturn, labelsToReturn

def proccessImage(image):
    image = cv.resize(image, (256,256))
    image = (image - 127.5) / 127.5
    return copy.deepcopy(image)

def segmentImage(image):
    images = []
    segmentCoordinates = []
    slidefilter(image, images, 1.2,segmentCoordinates)
    slidefilter(image, images, 1.5,segmentCoordinates)
    slidefilter(image, images, 1.5,segmentCoordinates) 
    slidefilter(image, images, 2,segmentCoordinates)
    slidefilter(image, images, 3,segmentCoordinates)
    slidefilter(image, images, 5, segmentCoordinates)
    return images, segmentCoordinates


def slidefilter(image, images, segmentQuoiant,segmentCoordinates):
    H, W, C = image.shape
    segmentHeight = math.floor(H/segmentQuoiant)
    segmentWidth = math.floor(W/segmentQuoiant)
    for i in range(segmentHeight, H ,75):
        for j in range(segmentWidth, W, 75) :
            temp_image = image[i-segmentHeight:i,j-segmentWidth:j]
            temp_image = cv.resize(temp_image, (256,256))
            temp_image = (temp_image - 127.5) / 127.5
            images.append(temp_image)

            segmentCoordinate = []
            segmentCoordinate.append( (segmentHeight,segmentWidth) )
            segmentCoordinate.append(j - segmentWidth)
            segmentCoordinate.append(i - segmentHeight)
            segmentCoordinates.append(segmentCoordinate)
    return segmentCoordinates

def relocateBbox(bbox, segmentCoordinates):
    segmentHeight = segmentCoordinates[0][0]
    segmentWidth = segmentCoordinates[0][1]
    x1 = int(bbox[0]*segmentWidth+segmentCoordinates[1])
    y1 = int(bbox[1]*segmentHeight+segmentCoordinates[2])
    x2 = int(bbox[2]*segmentWidth+segmentCoordinates[1])
    y2 = int(bbox[3]*segmentHeight+segmentCoordinates[2])
    return [x1,y1,x2,y2]


def splitImagesWithBoundingBox(filter_zero_lables=True, skip_data=0):
     # lists for data and labels
    images = []
    labels = []
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox_small.csv")
    ofile.readline()
    print("Starting to load data")
    if skip_data>0:
        for i in range(skip_data):
            ofile.readline()
    for line in ofile:
        bboxdata =line.split(",")
        x1_bbox = int(bboxdata[1])
        y1_bbox = int(bboxdata[2])
        x2_bbox = int(bboxdata[3])
        y2_bbox = int(bboxdata[4])

        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        H, W, C = image.shape
        one_sixth_image_height =  math.floor(H/8)
        one_sixth_image_width =  math.floor(W/8)
        segmentSize = min(math.floor(H/1.5),math.floor(W/1.5))
        segmentLocation = []
        image_segments = []

        for i in range(segmentSize, H ,one_sixth_image_height):
            for j in range(segmentSize, W, one_sixth_image_width  ) :
                temp_image = copy.deepcopy(image[i-segmentSize:i,j-segmentSize:j])

                temp_image = cv.resize(temp_image, (256,256))
                temp_image = (temp_image - 127.5) / 127.5

                segmentLocation.append( (j - segmentSize,i - segmentSize ) )
                image_segments.append(temp_image)

        boundingBoxModel = keras.models.load_model("MobileNet_bounding_box_model.hs")
        predictedBoundingBoxes = boundingBoxModel.predict(np.array(image_segments))

        for i in range(len(predictedBoundingBoxes)):

            x1_segment = predictedBoundingBoxes[i][0] * 255
            y1_segment = predictedBoundingBoxes[i][1] * 255
            x2_segment = predictedBoundingBoxes[i][2] * 255
            y2_segment = predictedBoundingBoxes[i][3] * 255


            for row in range(len(image_segments[i])):
                for column in range(len(image_segments[i][0])):
                    if column < x1_segment or row < y1_segment or column > x2_segment or row > y2_segment:
                        image_segments[i][row][column] = [0,0,0]

            predictedBoundingBox = relocateBbox(predictedBoundingBoxes[i],[segmentSize,segmentLocation[i][0],segmentLocation[i][1]])
            #coordinates of segment
            x1_fullimg = predictedBoundingBox[0]
            y1_fullimg = predictedBoundingBox[1]
            x2_fullimg = predictedBoundingBox[2]
            y2_fullimg = predictedBoundingBox[3]

            
            objectiveness_label = IoU([x1_bbox,y1_bbox,x2_bbox,y2_bbox],[x1_fullimg,y1_fullimg,x2_fullimg,y2_fullimg])
            labels.append(objectiveness_label)
            images.append(image_segments[i])
    # The data is biased toward some lables depending on the size of the segment, this gets rid of the bias in the data
    if filter_zero_lables:
        images,labels = filter_data(images,labels)
    # Simular data are also consecutive in the data, this shuffles it
    images, labels = shuffleData(images,labels)
    print(len(images))
    # break into train segment and test
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(labels[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(labels[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(labels[round(len(images)*0.9):])
    print("Finished loading data")
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)


def filterOutImages(images, bboxes):
     
    for i in range(len(bboxes)):

        x1_segment = bboxes[i][0] * 255
        y1_segment = bboxes[i][1] * 255
        x2_segment = bboxes[i][2] * 255
        y2_segment = bboxes[i][3] * 255


        for row in range(len(images[i])):
            for column in range(len(images[i][0])):
                if column < x1_segment or row < y1_segment or column > x2_segment or row > y2_segment:
                    images[i][row][column] = [0,0,0]
    return images

def segementImage(image):
    images = []
    images.append(image[:round(len(image)/2),:round(len(image)/2)])
    images.append(image[round(len(image)/2):,:round(len(image)/2)])
    images.append(image[:round(len(image)/2),round(len(image)/2):])
    images.append(image[round(len(image)/2):,round(len(image)/2):])
    return images