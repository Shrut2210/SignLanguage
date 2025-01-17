import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from moviepy.editor import *
import subprocess
import json
import yt_dlp

train_json = "../Data/Words/MSASL_train.json"

# Load the JSON data

def load_json_data(json_file):
    text, url = [], []
    with open(json_file, "r") as f:
        data = json.load(f)
    
    for item in data:
        text.append(item["text"])
        url.append(item["url"])
    return text, url

# Load the data
text, url = load_json_data(train_json)

save_path = "../Data/Words/MSASL_URL"
os.makedirs(save_path, exist_ok=True)

for item in url:
    try :
        title_command = [
            "yt-dlp",
            "--get-title",
            item
        ]
        
        result = subprocess.run(title_command, capture_output=True, text=True, check=True)
        video_title = result.stdout.strip()
        
        sanitized_title = "".join(c for c in video_title if c.isalnum() or c in " ._-").strip()

        video_save_path = os.path.join(save_path, sanitized_title)
        os.makedirs(video_save_path, exist_ok=True)
        
        download_command = [
                "yt-dlp",
                "-o", f"{video_save_path}/%(title)s.%(ext)s",
                item
            ]
        subprocess.run(download_command, check=True)
    except Exception as e:
        print(f"Error occurred while downloading video: {e}")
    

# capture_camera = cv2.VideoCapture(0)
# mp_drawings = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# while True:
#     ret, frame = capture_camera.read()
#     if not ret:
#         print("Error: Unable to read from the camera.")
#         break
    
#     RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(RGB_frame)
    
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             print(hand_landmarks)
    
#     cv2.imshow("Hand Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
    
# capture_camera.release()
# cv2.destroyAllWindows()
    