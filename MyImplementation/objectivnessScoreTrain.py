"""
Author: Troy Daniels
"""
import ModelFunctions
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# - objectiveness score - grab the data, set the model (have been changing this around)
(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages()

model = keras.Sequential([
       tf.keras.layers.Conv2D(24, kernel_size=(3,3), padding='same',activation='relu',
                           input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
    tf.keras.layers.Conv2D(40, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(42, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(120, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(100, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(150, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(100, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(200, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(100, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(400, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(200, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
# - objectiveness score - sometimes i use sigmoid output with binaryCrossentropy or no activation output with meansquaredloss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=tf.keras.losses.BinaryCrossentropy())
model.summary()
# tf.keras.utils.plot_model(model,to_file="my_model.png")
# - objectiveness score - save checkpoints where validation improves, fit model and produce loss graph
checkpoint_cb = keras.callbacks.ModelCheckpoint("objectivness_score_model.hs", save_best_only = True)
history = model.fit(training_data, training_labels, epochs=100, validation_data= (validation_data, validation_labels),callbacks = [checkpoint_cb])
pd.DataFrame(history.history).plot(ylim=(0, 0.1))
plt.show()
plt.savefig("objectivness_score_loss")
