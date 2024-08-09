import cv2
import mediapipe as mp
import numpy as np
import joblib

# Function to extract hand pose keypoints from an image using Mediapipe
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

# Load the trained MLP classifier model
mlp_model = joblib.load("C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_pose_classifier_model1.joblib")

# Load the new image
new_image_path = "C:/Users/dilip/PycharmProjects/Final-Year-Project/Dataset/Three/Three_53.jpg"

# Extract keypoints from the new image
new_hand_pose_keypoints = extract_connected_keypoints(new_image_path)
print(new_hand_pose_keypoints)
# Convert the keypoints to a NumPy array
new_hand_pose_keypoints_np = np.array(new_hand_pose_keypoints)

# Flatten the array of keypoints and reshape it to match the input shape used during training
new_flattened_keypoints = new_hand_pose_keypoints_np.flatten().reshape(1, -1)
# print(new_flattened_keypoints)
# Use the trained model to predict the label
predicted_label = mlp_model.predict(new_flattened_keypoints)

label_mapping = {
    'Five': 'Five',
    'Four': 'Four',
    'One': 'One',
    'Three': 'Three',
    'Two': 'Two'
}

predicted_gesture = label_mapping[predicted_label[0]]
print("Predicted gesture:", predicted_gesture)

# Get the ground truth label (assuming you have it)
ground_truth_label = "Three"  # Replace this with the actual ground truth label

# Check if the predicted label matches the ground truth label
if predicted_label == ground_truth_label:
    accuracy = 1.0  # If they match, accuracy is 100%
else:
    accuracy = 0.0  # If they don't match, accuracy is 0%

print("Accuracy:", accuracy)

