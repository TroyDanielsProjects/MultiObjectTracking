import tensorflow as tf
import ModelFunctions
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import pandas as pd

def build_model(input_shape):
    inputs= L.Input(input_shape)
    print(inputs.shape)


    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=0.75
    )


    x = backbone.output
    x = L.Conv2D(256,kernel_size=1,padding="same", activation='relu')(x)
    x = L.Flatten()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(1)(x)


    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    input_shape = (256,256,3)
    model = build_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    # tf.keras.utils.plot_model(model,to_file="moble_net_model.png")
    (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImagesWithBoundingBox()
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("MobileNet_myobjectiveness_score_model.hs", save_best_only = True)
    history = model.fit(training_data, training_labels, epochs=500, validation_data= (validation_data, validation_labels),callbacks = [checkpoint_cb])
    pd.DataFrame(history.history).plot(ylim=(0, 0.1))
    plt.show()
    plt.savefig("loss_myobjectiveness_graph_MobileNet_.png")
   