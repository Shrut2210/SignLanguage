import cv2
import os
import re

data_path = "../Data/Words/downloaded_videos"
save_path = "../Data/Words/downloaded_images"
os.makedirs(save_path, exist_ok=True)

for word in os.listdir(data_path):
    word_path = os.path.join(data_path, word)
    word = word.lower()
    word = re.sub(r'[^a-zA-Z0-9]', '_', word)
    word_save_path = os.path.join(save_path, word)
    os.makedirs(word_save_path, exist_ok=True)
    
    for video_file in os.listdir(word_path):
        if video_file.endswith(".mp4") :
            video_path = os.path.join(word_path, video_file)
        
            # Create a directory for the frames of the current video
            # os.makedirs(f"downloaded_images/{word}", exist_ok=True)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            c_f = 0  # Frame counter
            while True:
                ret, frame = cap.read()
                
                # Break the loop if the video has ended
                if not ret:
                    if c_f == 0:
                        print(f"Error: Unable to open video or no frames to read for {video_file}.")
                    else:
                        print(f"End of video file reached for {video_file}.")
                    break
                
                # Save the current frame as an imag
                name = f'../Data/Words/downloaded_images/{word}/{word}_{c_f}.jpg'
                
                cv2.imwrite(name, frame)
                
                print(f"Saved frame {c_f} as {name}")
                
                c_f += 1
            
            # Release video capture object and close OpenCV windows
            cap.release()
cv2.destroyAllWindows()