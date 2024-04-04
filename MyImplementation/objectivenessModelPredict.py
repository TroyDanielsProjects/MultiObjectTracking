import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import testBBRData


(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages(skip_data=600)


# - objectiveness score - make prediction for the test data (I am skipping most of the data so I can load it locally, all is actually part of validation (within 90% of data))
def seeObjectivenessPrediction(testing_data, testing_labels, model_name = "MobileNet_objectiveness_model_full_distributed_data.hs"):
    model = keras.models.load_model(model_name)
    y_pred = model.predict(testing_data)
    distance = 0
    for i in range(len(testing_data)):
        distance += abs(y_pred[i][0] - testing_labels[i])
    aveDistance = distance / len(testing_data)
    print("The average error (distance) over testing is: " + str(aveDistance))

seeObjectivenessPrediction(training_data,training_labels)