import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn as nn
import time

# Define the test transforms (same as during training)
mean = [0.5187, 0.4988, 0.5147]
std = [0.2017, 0.2310, 0.2390]
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# # Load the saved model
# checkpoint_path = 'best_model.pth'
# try:
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
# except FileNotFoundError:
#     print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
#     exit()

# Reinitialize the model
num_classes = 26  # Number of classes in your dataset
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the checkpoint
checkpoint_path = 'best_model.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    if isinstance(checkpoint, dict):  # If it's a state_dict
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, torch.nn.Module):  # If it's the full model
        model = checkpoint
    else:
        raise TypeError("Unexpected checkpoint format")
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
    exit()
except TypeError as e:
    print(f"Error loading model: {e}")
    exit()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mapping class indices to labels (example mapping; update according to your dataset)
class_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Function to preprocess the frame and predict the label
def predict_frame(frame, model):
    # Convert the frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the transformations
    image_tensor = test_transforms(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return class_map[predicted_class.item()], confidence.item()

# Start the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit.")
fps = 0.0
start_time = time.time()

# Log predictions to a file
log_file = open("predictions_log.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Predict the label for the current frame
    label, confidence = predict_frame(frame, model)
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    start_time = time.time()
    
    # Display the prediction and confidence on the frame
    cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Live ASL Recognition", frame)
    
    # Log the prediction to the file
    log_file.write(f"Prediction: {label}, Confidence: {confidence:.2f}, FPS: {fps:.2f}\n")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
log_file.close()
