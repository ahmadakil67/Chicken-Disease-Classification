import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def predict(self):

        # Load trained model
        model_path = os.path.join(
            "artifacts",
            "training",
            "model.h5"
        )

        model = load_model(model_path)


        # Load image
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )


        # Convert image to array
        test_image = image.img_to_array(test_image)


        # Apply the SAME preprocessing used during training
        test_image = tf.keras.applications.vgg16.preprocess_input(
            test_image
        )


        # Add batch dimension
        test_image = np.expand_dims(
            test_image,
            axis=0
        )


        # Prediction probabilities
        predictions = model.predict(
            test_image,
            verbose=0
        )


        # Predicted class index
        predicted_class = np.argmax(
            predictions,
            axis=1
        )[0]


        # Class mapping
        class_names = {
            0: "Coccidiosis",
            1: "Healthy",
            2: "Salmonella"
        }


        prediction = class_names[
            predicted_class
        ]


        print(
            "Predicted class:",
            prediction
        )


        return [
            {
                "image": prediction
            }
        ]