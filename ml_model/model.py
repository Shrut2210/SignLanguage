import pandas as pd
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import seaborn as sns
import cv2
import os

data_path = os.path.join("../Data")
train_data_path = os.path.join(data_path, "asl_alphabet_train/asl_alphabet_train")
test_data_path = os.path.join(data_path, "asl_alphabet_test/asl_alphabet_test")

img_size = 64
classes = sorted(os.listdir(train_data_path))
num_classes = len(classes)

print(classes.index('A'))

def load_train_data(data_path):
    images, labels = [], []
    for label in os.listdir(data_path):
        class_path = os.path.join(data_path, label)
        if os.path.isdir(class_path): 
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):  
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        continue
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(classes.index(label))  
    return np.array(images), np.array(labels)

# def load_test_data(data_path):
#     images, labels = [], []
#     if os.path.isdir(data_path):  
#         for img_file in os.listdir(data_path):
#             img_path = os.path.join(data_path, img_file)
#             img = cv2.imread(img_path)
#             if img is None:  
#                 print(f"Failed to load image: {img_path}")
#                 continue  
#             img = cv2.resize(img, (img_size, img_size))  
#             images.append(img)
#             labels.append(img_file)
#     return images, labels


train_images, train_labels = load_train_data(train_data_path)
# test_images, test_labels = load_test_data(test_data_path)

train_images = train_images / 255.0
# test_images /= 255.0

train_labels = to_categorical(train_labels, num_classes)
# test_labels = to_categorical(test_labels, num_classes)

# label_to_index = {label : idx for idx, label in enumerate(classes)}
# numerical_classes = [label_to_index[label] for label in train_labels]

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)

model.save('asl_model.h5')