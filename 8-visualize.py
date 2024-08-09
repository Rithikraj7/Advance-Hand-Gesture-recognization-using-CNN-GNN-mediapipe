import cv2
import mediapipe as mp
import numpy as np
import os

# Function to extract connected keypoints from an image
def extract_connected_keypoints(image_path):
    # Load the image with landmarks drawn on it
    image_with_landmarks = cv2.imread(image_path)
    # Convert the image to RGB format if it's not already in that format
    image_with_landmarks_rgb = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    # Process the image to extract hand pose landmarks
    results = hands.process(image_with_landmarks_rgb)
    # Define connections between landmarks
    connections = [
        # Thumb
        (1, 2), (2, 3), (3, 4), (4, 5),
        # Index finger
        (5, 6), (6, 7), (7, 8), (8, 9),
        # Middle finger
        (9, 10), (10, 11), (11, 12), (12, 13),
        # Ring finger
        (13, 14), (14, 15), (15, 16), (16, 17),
        # Little finger
        (17, 18), (18, 19), (19, 20), (20, 0),
        # Connections between fingers
        (0, 1), (0, 0), (0, 0), (0, 0)
    ]

    # Extract landmarks if hand is detected
    connected_keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for connection in connections:
                kp1_index, kp2_index = connection
                kp1 = hand_landmarks.landmark[kp1_index]
                kp2 = hand_landmarks.landmark[kp2_index]
                connected_keypoints.append([kp1.x, kp1.y, kp1.z])
                connected_keypoints.append([kp2.x, kp2.y, kp2.z])
    return connected_keypoints

# Example usage
image_path = "C:/Users/dilip/PycharmProjects/Final-Year-Project/Dataset/Four/Four_71.jpg"
connected_keypoints = extract_connected_keypoints(image_path)
print("Connected keypoints:", connected_keypoints)



# Define the image size
image_width = 640
image_height = 480

# Create a blank image
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Define the color for drawing connections (in BGR format)
connection_color = (255, 0, 0)  # Blue color

# Draw lines between connected keypoints
for i in range(len(connected_keypoints) - 1):
    start_point = (int(connected_keypoints[i][0] * image_width), int(connected_keypoints[i][1] * image_height))
    end_point = (int(connected_keypoints[i+1][0] * image_width), int(connected_keypoints[i+1][1] * image_height))
    cv2.line(image, start_point, end_point, connection_color, 2)

# Show the image with connected keypoints
cv2.imshow("Connected Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
