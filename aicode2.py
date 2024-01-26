import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def load_and_preprocess_model(model_path):
    """
    Load the pre-trained model and return the model instance.
    """
    model = load_model(model_path)
    return model

def preprocess_image(image_path):
    """
    Load and preprocess the image for model prediction.
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array

def predict_image_class(model, img_array):
    """
    Predict the class of the input image using the pre-trained model.
    """
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    return class_index, confidence

def get_class_label(class_index):
    """
    Map class index to a human-readable label.
    """
    class_labels = {0: 'Class_0', 1: 'Class_1', 2: 'Class_2'}  # Add more labels if needed
    return class_labels.get(class_index, 'Unknown')

# Get the full path to the pre-trained model
model_path = "./image_classifier_model.h5"  # Update with the correct path

# Load the pre-trained model
loaded_model = load_and_preprocess_model(model_path)

# Placeholder image path (replace with the actual image path when testing)
image_path = "./test_image.jpg"

# Preprocess the image
img_array = preprocess_image(image_path)

# Make predictions
class_index, confidence = predict_image_class(loaded_model, img_array)
predicted_label = get_class_label(class_index)

# Display the results
print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence:.2%}")
