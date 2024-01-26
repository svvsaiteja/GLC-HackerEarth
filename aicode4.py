import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Set the path to your dataset
dataset_dir = "Data"

# Define the parameters
input_shape = (224, 224, 3)
num_classes = 3
batch_size = 32
epochs = 15  # Adjusted number of epochs

# Create a CNN model using ResNet50 as a base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Fine-tune the last few layers of the base model
for layer in base_model.layers[:-10]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Adjusted number of neurons
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Image data augmentation with increased intensity
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model with increased number of epochs
history = model.fit(training_set, epochs=epochs)

# Save and load the model as before...

# Make predictions on new images as before...
