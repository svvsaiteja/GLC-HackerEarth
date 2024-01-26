import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Set the path to your dataset
dataset_dir = "Data"

# Define the parameters
input_shape = (224, 224, 3)
num_classes = 3
batch_size = 32
epochs = 10

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

# Train the model
model.fit(training_set, epochs=epochs)

# Save the model
model.save("image_classifier_model.h5")

# Load the saved model
loaded_model = tf.keras.models.load_model("image_classifier_model.h5")

# Make predictions on new images
img_path = "./test_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Make predictions
predictions = loaded_model.predict(img_array)
class_index = tf.argmax(predictions[0]).numpy()

# Print the predicted class index
print(f"Predicted class index: {class_index}")
