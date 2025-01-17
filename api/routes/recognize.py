import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('../../new_model/asl_model.h5')

# Constants
img_size = 32
classes = sorted(['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    # Check if a hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            
            # Expand the bounding box slightly
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Check if the bounding box is valid
            if x_max > x_min and y_max > y_min:
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                # Ensure the cropped region is not empty
                if hand_region.size > 0:
                    # Preprocess the hand region
                    hand_region = cv2.resize(hand_region, (img_size, img_size))
                    hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
                    hand_region_norm = hand_region_rgb / 255.0
                    hand_region_reshape = np.expand_dims(hand_region_norm, axis=0)
                    
                    # Predict the class
                    predictions = model.predict(hand_region_reshape)
                    
                    # Check predictions validity
                    # if len(predictions[0]) != len(classes):
                    #     print(f"Warning: Model predictions ({len(predictions[0])}) do not match classes ({len(classes)}).")
                    #     continue
                    
                    predicted_index = np.argmax(predictions)
                    
                    # Validate the predicted index
                    if 0 <= predicted_index < len(classes):
                        predicted_class = classes[predicted_index-1]
                        confidence_score = predictions[0][predicted_index] * 100
                        
                        # Draw the bounding box and prediction
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, f"{predicted_class} ({confidence_score:.2f}%)", 
                                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        print(f"Invalid predicted index: {predicted_index}")
    
    # Display the frame
    cv2.imshow('ASL Sign Prediction', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()