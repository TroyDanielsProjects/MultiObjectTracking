import cv2 as cv
import random
import copy
import ModelFunctions
import numpy as np
import tensorflow as tf
from tensorflow import keras
import testBBRData

def see_data():
    # lists for data and labels
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    print("Starting to load data")
    count = 0
    images = []
    origImages = []
    bboxes = []
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
        origImage = copy.deepcopy(image)
        image = cv.resize(image, (256,256))
        image = (image - 127.5)/127.5
        origImages.append(origImage)
        images.append(image)
        norm_x1 = x1 / W
        norm_y1 = y1 / H
        norm_x2 = x2 / W
        norm_y2 = y2 / H
        # if count < 11 and count >5:
        #     resized_x1 = norm_x1 * 256
        #     resized_y1 = norm_y1 * 256
        #     resized_x2 = norm_x2 * 256
        #     resized_y2 =norm_y2 * 256
        bbox = [norm_x1,norm_y1,norm_x2,norm_y2]
        bboxes.append(bbox)
        # add the data/label to the lists
        # if  count < 5:
        #     testBBRData.showBoundingBoxWithCoordinates(origImage,[x1,y1,x2,y2])
        # elif count < 11 and count > 5:
        #     testBBRData.showBoundingBoxWithCoordinates(image,[resized_x1,resized_y1,resized_x2,resized_y2], fullimage=False)
        # count+=1
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(bboxes[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(bboxes[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(bboxes[round(len(images)*0.9):])
    print("Finished loading data")
    myModel = keras.models.load_model("my_bounding_box_model")
    mobileNetModel = keras.models.load_model("MobileNet_bounding_box_model.hs")
    yPredMyModel = myModel.predict(testing_data)
    
    yPredMobileNetModel = mobileNetModel.predict(testing_data)
    for i in range(len(testing_data)):
        # testBBRData.seeBothBoundingBoxes(testing_data[i],testing_labels[i],yPredMyModel[i],yPredMobileNetModel[i])
        ModelFunctions.segementImage(testing_data[i])


def segments():
    # lists for data and labels
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    print("Starting to load data")
    mobileNetModel = keras.models.load_model("MobileNet_bounding_box_model.hs")
    count = 0
    images = []
    origImages = []
    bboxes = []
    ofile.readline()
    ofile.readline()
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
        origImage = copy.deepcopy(image)
        image = cv.resize(image, (256,256))
        image = (image - 127.5)/127.5
        origImages.append(origImage)
        images.append(image)
        norm_x1 = x1 / W
        norm_y1 = y1 / H
        norm_x2 = x2 / W
        norm_y2 = y2 / H
        segmentImages = ModelFunctions.segementImage(origImage)
        for i in range(len(segmentImages)):
            segmentImages[i] = cv.resize(segmentImages[i], (256,256))
            segmentImages[i] = (segmentImages[i] - 127.5)/127.5
        segmentImages = np.array(segmentImages)
        yPredMobileNetModel = mobileNetModel.predict(segmentImages)

        yPredMobileNetModel[0][0] = round(yPredMobileNetModel[0][0] * W/2 )
        yPredMobileNetModel[0][1] = round(yPredMobileNetModel[0][1] * H/2 )
        yPredMobileNetModel[0][2] = round(yPredMobileNetModel[0][2] * W/2 )
        yPredMobileNetModel[0][3] = round(yPredMobileNetModel[0][3] * H/2 )

        yPredMobileNetModel[1][0] = round(yPredMobileNetModel[1][0] * W/2 + W/2)
        yPredMobileNetModel[1][1] = round(yPredMobileNetModel[1][1] * H/2 )
        yPredMobileNetModel[1][2] = round(yPredMobileNetModel[1][2] * W/2 + W/2)
        yPredMobileNetModel[1][3] = round(yPredMobileNetModel[1][3] * H/2 )

        yPredMobileNetModel[2][0] = round(yPredMobileNetModel[2][0] * W/2 )
        yPredMobileNetModel[2][1] = round(yPredMobileNetModel[2][1] * H/2 + H/2)
        yPredMobileNetModel[2][2] = round(yPredMobileNetModel[2][2] * W/2 )
        yPredMobileNetModel[2][3] = round(yPredMobileNetModel[2][3] * H/2 +H/2)

        yPredMobileNetModel[3][0] = round(yPredMobileNetModel[3][0] * W/2 + W/2)
        yPredMobileNetModel[3][1] = round(yPredMobileNetModel[3][1] * H/2 + H/2)
        yPredMobileNetModel[3][2] = round(yPredMobileNetModel[3][2] * W/2 + W/2)
        yPredMobileNetModel[3][3] = round(yPredMobileNetModel[3][3] * H/2 + H/2)
            
        testBBRData.showBoundingBoxes(origImage,yPredMobileNetModel)
segments()