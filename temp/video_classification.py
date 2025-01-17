from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

data_path = os.path.join("../Data")
train_data_path = os.path.join(data_path, "Words/downloaded_videos")

labels = []

for label in os.listdir(train_data_path):
    all_labels = os.path.join(train_data_path, label)
    
    for label_file in os.listdir(all_labels):
        if label_file.endswith(".mp4"):
            labels.append({'tag': label , 'name' : os.path.join(all_labels, label_file)})
        
train_df = pd.DataFrame(labels)

# print(train_df.head())

df = train_df.loc[:, ['name', 'tag']]
df.to_csv("train_data.csv")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
    

train_df = pd.read_csv("train_data.csv")
# print(len(train_df ))

IMG_FILE = 224
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def crop_center_square(frame) :
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x - min_dim) // 2
    start_y = (y - min_dim) // 2
    
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_FILE, IMG_FILE)):
    cap = cv2.VideoCapture(path)
    frames = []
    
    try :
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2,1,0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally :        
        cap.release()
    return np.array(frames)

def build_feature_extractor() :
    feature_extractor = keras.applications.InceptionV3(
        weights = "imagenet",
        include_top = False,
        input_shape = (IMG_FILE, IMG_FILE, 3),
        pooling = 'avg'
    )    
    
    preprocess_input = keras.applications.inception_v3.preprocess_input
    
    inputs = keras.Input((IMG_FILE, IMG_FILE, 3))
    preprocessed = preprocess_input(inputs)
    
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
# print(label_processor.get_vocabulary())

labels_data = train_df["tag"].values
labels_data = label_processor(labels_data[..., None]).numpy()
# print(labels_data)

def prepare_videos(df, root_dir) :
    num_samples = len(df)
    video_paths = df["name"].values.tolist()
    
    labels = df['tag'].values
    labels = label_processor(labels[..., None]).numpy()
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for idx, path in enumerate(video_paths):
        print(path)
        frames = load_video(path)
        frames = frames[None, ...]

        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1 

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

train_data, train_labels = prepare_videos(train_df, data_path)

print(train_data[0].shape, train_labels.shape)

def get_seq_model() :
    class_vocab = label_processor.get_vocabulary()
    
    frame_feature_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    