# import numpy as np
# import pandas as pd
# import os
# import json
# from moviepy.editor import *
# from pytube import YouTube, Playlist
# import re
# import subprocess
# import yt_dlp

# train_json = "../Data/Words/MSASL_train_1.json"

# def load_json_data(json_file):
#     text, url = [], []
#     with open(json_file, "r") as f:
#         data = json.load(f)
    
#     for item in data:
#         text.append(item["text"])
#         url.append(item["url"])
#     return text, url

# # Load the data
# label, url = load_json_data(train_json)
# save_data = "../Data/Words/MSASL/videos"

# os.makedirs(save_data, exist_ok=True)

# for item in url:
#     try :
#         title_command = [
#             "yt-dlp",
#             "--get-title",
#             item
#         ]
        
#         index = url.index(item)
        
        
#         result = subprocess.run(title_command, capture_output=True, text=True, check=True)
        
#         video_save_path = os.path.join(save_data, label[index])
#         if(os.path.exists(video_save_path)):
#             continue
        
#         os.makedirs(video_save_path, exist_ok=True)
        
#         download_command = [
#             "yt-dlp",
#             "-o", f"{video_save_path}/%(lambda_1[index])s.%(ext)s",
#             item
#         ]
        
#         subprocess.run(download_command, check=True)
        
#         print(f"Downloaded: {label[index]} to {video_save_path}")
#     except Exception as e:
#         print(f"Unexpected error for {item}: {e}")

import torch

print(torch.cuda.is_available())
# import os

# data_path = os.path.join("../Data/Words/downloaded_images")

# for word in os.listdir(data_path):
    
#     word_path = os.path.join(data_path, word)
    
#     images = os.listdir(word_path)

#     sorted_images = sorted(
#         images,
#         key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.startswith("${word}_") and x.endswith('.jpg') else float('inf')
#     )
    
#     print(sorted_images)

#     if len(sorted_images) > 10:
#         start_index = 10
#         end_index = len(sorted_images) - 10
#         for i in range(0, start_index):
#             image_path = os.path.join(word_path, sorted_images[i])
#             os.remove(image_path)
        
#         for i in range(end_index, len(sorted_images)):
#             image_path = os.path.join(word_path, sorted_images[i])
#             os.remove(image_path)
        
#         print(f"Removed first and last 10 images from", word)
    
    