"""
Author: Troy Daniels

This file was created to check the data. The first method is to see the general sizing and format of the images. 
The second is to visualize the image with its corresponding bbox
"""

import cv2 as cv
import random
import copy
import ModelFunctions

def getImageInfor():
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    max_height = 0
    max_width = 0
    max_size = 0
    min_height = None
    min_width = None
    min_size = None
    ave_height = 0
    ave_width = 0
    ave_size = 0
    count = 0
    for line in ofile:
        bboxdata =line.split(",")
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        img = cv.imread(imgPath)
        h,w,c = img.shape
        size = h * w

        ave_height+=h
        ave_width+=w
        ave_size+= size
        count+=1

        if h>max_height:
            max_height = h
        if min_height == None or h<min_height:
            min_height = h

        if w>max_width:
            max_width = w
        if min_width == None or w<min_width:
            min_width = w

        if size>max_size:
            max_size = size
        if min_size == None or size<min_size:
            min_size = size
    ave_height /= count
    ave_width /= count
    ave_size /= count
    print(f'the max height is {max_height}\n the min height is {min_height}\n the ave height is {ave_height}\n the max width is {max_width}\n the min width is {min_width}\n the average width is {ave_width}\n the max size is {max_size}\n the min size is {min_size}\n the ave size is {ave_size}')
"""
The answer I got above is - 
the max height is 4752
 the min height is 1049
 the ave height is 2716.3612750885477
 the max width is 5184
 the min width is 1440
 the average width is 2571.219598583235
 the max size is 17915904
 the min size is 2073600
 the ave size is 6991014.002361275

this is to see the data, see the image with the bounding box (label drawn)
"""

def seeAlldata():
    ofile = open("./archive/bbox.csv")
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
        cv.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 5)
        cv.imshow('test', image)
        cv.waitKey(0)
        cv.destroyAllWindows()



"""

This is to see a visual representation of the sliding window approch

"""

def seeImageSegemts(images, labels):
    for i in range(len(images)):
        print(labels[i])
        cv.imshow('test', images[i])
        cv.waitKey(0)
        cv.destroyAllWindows()

def slidingWindowVisualization():
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    for line in ofile:
        bboxdata =line.split(",")
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        image = cv.resize(image, (500,500))
        for i in range(0,500,200):
            for j in range(0,500,200):
                cv.rectangle(image, (j,i), (j+200,i+200), (i,j,i+j), 2)
        cv.imshow('test', image)
        cv.waitKey(0)
        cv.destroyAllWindows()

def seeBoundingBoxes(image,true_bbox,pred_bbox):
    x1_true = int(true_bbox[0]*255)
    y1_true = int(true_bbox[1]*255)
    x2_true = int(true_bbox[2]*255)
    y2_true = int(true_bbox[3]*255)
    x1_pred = int(pred_bbox[0]*255)
    y1_pred = int(pred_bbox[1]*255)
    x2_pred = int(pred_bbox[2]*255)
    y2_pred = int(pred_bbox[3]*255)
    cv.rectangle(image, (x1_true,y1_true), (x2_true,y2_true), (255,0,0), 5) # this is blue for true
    cv.rectangle(image, (x1_pred,y1_pred), (x2_pred,y2_pred), (0,255,0), 5) # this is green for pred
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def saveBoundingBoxes(image,bboxes,path,img_name):
    image = copy.deepcopy(image)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        randColor1 = random.randrange(0,255)
        randColor2 = random.randrange(0,255)
        randColor3 = random.randrange(0,255)
        cv.rectangle(image, (x1,y1), (x2,y2), (randColor1,randColor2,randColor3), 15) # this is blue for true
    cv.imwrite(path + img_name +".jpg", image)


def showBoundingBoxes(image,bboxes):
    image = copy.deepcopy(image)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        randColor1 = random.randrange(0,255)
        randColor2 = random.randrange(0,255)
        randColor3 = random.randrange(0,255)
        cv.rectangle(image, (x1,y1), (x2,y2), (randColor1,randColor2,randColor3), 5) # this is blue for true
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showBoundingBox(image,bbox):
    image = copy.deepcopy(image)
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    cv.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 5)
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showBoundingBoxWithCoordinates(image,bbox, fullimage=True):
    image = copy.deepcopy(image)
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    H, W, C = image.shape
    cv.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 3)
    if fullimage:
        cv.circle(image, (x1,y1), 10, (0,255,0), 10)
        cv.circle(image, (x2,y2), 10, (0,255,0), 10)
        cv.putText(image, f"(X1,Y1): ({x1},{y1})", (0,50), cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
        cv.putText(image, f"(X2,Y2): ({x2},{y2})", (0,100), cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255),3)
    else:
        cv.putText(image, f"(X1,Y1): ({x1/256},{y1/256})", (0,50), cv.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),3)
        cv.putText(image, f"(X2,Y2): ({x2/256},{y2/256})", (0,100), cv.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),3)
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showBoundingBoxsWithPixelError(image,bbox,pred):
    image = copy.deepcopy(image)
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    x1_pred = int(pred[0])
    y1_pred = int(pred[1])
    x2_pred = int(pred[2])
    y2_pred = int(pred[3])


    cv.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 5)
    cv.rectangle(image, (x1_pred,y1_pred), (x2_pred,y2_pred), (0,255,0), 5)
    cv.putText(image, f"Ground Truth", (0,0), cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
    cv.putText(image, f"Ground Truth", (0,10), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0),3)
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def seeBothBoundingBoxes(image,true_bbox,pred_bbox,mobile):
    H, W, C = image.shape
    x1_true = int(true_bbox[0]*W)
    y1_true = int(true_bbox[1]*H)
    x2_true = int(true_bbox[2]*W)
    y2_true = int(true_bbox[3]*H)
    x1_pred = int(pred_bbox[0]*W)
    y1_pred = int(pred_bbox[1]*H)
    x2_pred = int(pred_bbox[2]*W)
    y2_pred = int(pred_bbox[3]*H)
    x1_mobile = int(mobile[0]*W)
    y1_mobile = int(mobile[1]*H)
    x2_mobile = int(mobile[2]*W)
    y2_mobile = int(mobile[3]*H)
    cv.rectangle(image, (x1_true,y1_true), (x2_true,y2_true), (255,0,0), 5) # this is blue for true
    cv.rectangle(image, (x1_pred,y1_pred), (x2_pred,y2_pred), (0,255,0), 5) # this is green for pred
    cv.rectangle(image, (x1_mobile,y1_mobile), (x2_mobile,y2_mobile), (0,0,255), 5) # this is green for pred
    cv.putText(image, f"Ground Truth Bounding Box", (0,0), cv.FONT_HERSHEY_SIMPLEX,1, (255,0,0),3)
    cv.putText(image, f"My Models Bouning Box", (0,50), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0),3)
    cv.putText(image, f"MobileNet Bouning Box", (0,100), cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()