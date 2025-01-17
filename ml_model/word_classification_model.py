import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define paths
data_path = os.path.join("../Data")
train_data_path = os.path.join(data_path, "Words/downloaded_images")

# Hyperparameters
img_size = 64
batch_size = 128
num_epochs = 30
num_classes = 2424

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

model = Sequential()

model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2424, activation='softmax'))


# # Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = os.path.join(data_path, "Log")

tensor_callback = TensorBoard(log_dir=log_dir)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=30,
    callbacks=[tensor_callback]
)

model_json = model.to_json()

with open("word_asl_model.json", "w") as json_file:
    json_file.write(model_json)

model.save("word_asl_model.keras")

# # # Evaluate on test data
# # test_loss, test_accuracy = model.evaluate(test_images, test_labels, batch_size=batch_size)
# # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # # Plot training history
# # plt.figure(figsize=(12, 4))
# # plt.subplot(1, 2, 1)
# # plt.plot(history.history['accuracy'], label='Training Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # plt.legend()
# # plt.title('Accuracy Over Epochs')

# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['loss'], label='Training Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.legend()
# # plt.title('Loss Over Epochs')

# # plt.show()