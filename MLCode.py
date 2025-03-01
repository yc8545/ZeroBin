pip install tensorflow numpy matplotlib opencv-python scikit-learn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/WasteClassificationDataset"

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

# Check if paths exist
import os

for folder in [train_dir, val_dir, test_dir]:
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
    else:
        print(f"✅ Folder found: {folder}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data augmentation for training images
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    "/content/drive/MyDrive/WasteClassificationDataset/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    "/content/drive/MyDrive/WasteClassificationDataset/val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    "/content/drive/MyDrive/WasteClassificationDataset/test",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

model = keras.Sequential([
    keras.layers.Input(shape=(150,150,3)),  # Define input explicitly
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')  # 3 Classes: Blue, Green, Red
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

model.save('/content/drive/My Drive/waste_classifier_model.keras')

from tensorflow import keras

# Load the model
model = keras.models.load_model('/content/drive/My Drive/waste_classifier_model.keras')
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model loaded successfully!")
