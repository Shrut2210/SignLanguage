import numpy as np
import pandas as pd
import os
import json
from moviepy.editor import *
from pytube import YouTube, Playlist
import re
import subprocess
import yt_dlp

url_list = [
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5F7O9tqdL4xNiYmO672Hhl",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6ZxJtFc5D-2hDRcaDTQRe1",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6xaoKcbYf6h0vh1VSPCkAn",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD72hDGHXqelYRJ24v_0YLTc",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5ssKJYhdcf05I_fTvH7NK8",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4dx2WI89UvsailNA6HbJ3_",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD66_9VMfTL9VcGpsQw0OSn8",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6HNL11BMZyIK09HQajCBO3",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4MsRMBuYnGIqmEjy758ck2",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6XX7ZIXeis4u-rIEUJ05VV",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4VrAbAX1torpgGg2-4StnK",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5PXMj4kGFCbdMrdDyJNdXU",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6_JAkIo1Y6RX3V1xmK2SmN",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD7QJWUGoyU1DOv5hhRhxsPy",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6U9JtWubZrXZW-eC6OaEap",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6B5GkGo0Ej-tpaImaiuZeA",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5g-ApCSdr5a2GPUSINiA5I",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5eNdPGSVuynJpQQx-1p4c_",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD675DArnwB4tGQ_lkiMG_f3",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4jXNMPi3Lw2MHLj6tMs6Yi",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5aWwP3EH6rN9QMUcLC-8Kn",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD7uu2sxAMBXeKGDaxAGvo2G",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5sAIWCdog_XaHmLUdhtk-K",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5gA1K4kPiFVSqDULhMWkfW",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD7PDBVuseS6Qbyz-lMWby_s",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4-Ysg0CYHmOU7W6WcuLp7W",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD45V1HyKIx0eZ9SQGcJpEvL",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4em5s6JqlC5Y0LqL66nP1O",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6sd0rduSNIeB7IuXAeOzS_",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6zXC8Qkcqnyr2SZFmZpk95",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4vkT4uL_-KB_bm1UAyBk31",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6JeUqXaIJowcWzVQUrrfsg",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4wdhfmMnBKTcBln8qwmw1J",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD6mu3Uu0yxRzuh9BvoPBYT2",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD4-r2hVKtmGmWnfruMxhWmG",
    "https://www.youtube.com/playlist?list=PLoSZTuVrsuD5RxSb1JosElpxVnW44mr8X"
]

save_path = "../Data/Words/downloaded_videos"
os.makedirs(save_path, exist_ok=True)

for url_link in url_list :
    pl = Playlist(url_link)

    for video_url in pl.video_urls:
        try:
            title_command = [
                "yt-dlp",
                "--get-title",
                video_url
            ]
            result = subprocess.run(title_command, capture_output=True, text=True, check=True)
            video_title = result.stdout.strip()

            sanitized_title = "".join(c for c in video_title if c.isalnum() or c in " ._-").strip()

            video_save_path = os.path.join(save_path, sanitized_title)
            os.makedirs(video_save_path, exist_ok=True)

            download_command = [
                "yt-dlp",
                "-o", f"{video_save_path}/%(title)s.%(ext)s",
                video_url
            ]
            subprocess.run(download_command, check=True)

            print(f"Downloaded: {video_title} to {video_save_path}")
        except Exception as e:
            print(f"Unexpected error for {video_url}: {e}")