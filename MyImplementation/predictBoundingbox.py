import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import testBBRData
import math


(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.load_data()


def IoUMetric(testing_data, testing_labels, model_name = "my_bounding_box_model"):
    model = keras.models.load_model(model_name)
    average_IoU = 0
    y_pred = model.predict(testing_data)
    for i in range(len(testing_data)):
        for j in range(4):
            if y_pred[i][j] > 1:
                y_pred[i][j] = 1.0
            elif y_pred[i][j] < 0:
                y_pred[i][j] = 0
        average_IoU += ModelFunctions.IoU(testing_labels[i], y_pred[i])
    average_IoU/=len(y_pred)
    standardDeviation = 0
    for i in range(len(testing_data)):
        standardDeviation = (ModelFunctions.IoU(testing_labels[i], y_pred[i]) - average_IoU) ** 2
    standardDeviation/=len(y_pred)
    standardDeviation = math.sqrt(standardDeviation)
    print(f'The average intersection over union is {average_IoU} with the Standard deviation of {standardDeviation}')

def seePredBbox(testing_data, testing_labels, model_name = "my_bounding_box_model"):
    model = keras.models.load_model(model_name)
    y_pred = model.predict(testing_data)
    print(y_pred)
    for i in range(len(testing_data)):
        testBBRData.seeBoundingBoxes(testing_data[i],testing_labels[i],y_pred[i])

def seeBothModels(testing_data,testing_labels):
    model = keras.models.load_model("my_bounding_box_model")
    mobileNet = keras.models.load_model("my_bounding_box_model")
    y_pred = model.predict(testing_data)
    y_mobile = mobileNet.predict(testing_data)
    for i in range(len(testing_data)):
        testBBRData.seeBothBoundingBoxes(testing_data[i],testing_labels[i],y_pred[i],y_mobile[i])



seeBothModels(testing_data,testing_labels)

# print("For my model")
# IoUMetric(testing_data, testing_labels)
# seePredBbox(testing_data, testing_labels, model_name="MobileNet_bounding_box_model.hs")
# print("For pre-set model")
# # model_name="resources/MobileNet_bounding_box_model.hs"
# IoUMetric(testing_data, testing_labels,model_name="MobileNet_bounding_box_model.hs")