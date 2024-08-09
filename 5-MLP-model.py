# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# from sklearn.neural_network import MLPClassifier
#
# # Function to extract hand pose keypoints from an image
# def extract_keypoints(image_path):
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands()
#     results = hands.process(image_rgb)
#     hand_pose_keypoints = []
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = landmark.x
#                 y = landmark.y
#                 z = landmark.z  # Depth value (optional, if available)
#                 hand_pose_keypoints.append((x, y, z))
#     return np.array(hand_pose_keypoints)
#
# # Path to the dataset folder
# dataset_folder = "C:/Users/dilip/PycharmProjects/final-code-project/GCN+CNN/Dataset"
#
# # Initialize lists to store data
# X_data = []
# y_data = []
#
# # Iterate through each folder in the dataset
# for folder in os.listdir(dataset_folder):
#     folder_path = os.path.join(dataset_folder, folder)
#     if os.path.isdir(folder_path):
#         # Initialize a counter to track the number of images processed for each class
#         count = 0
#         # Iterate through each image in the folder
#         for filename in os.listdir(folder_path):
#             # Check if the count exceeds 10 for the current class
#             if count >= 10:
#                 break  # Break the loop if 10 images have been processed
#             image_path = os.path.join(folder_path, filename)
#             # Extract keypoints and append to the data lists
#             keypoints = extract_keypoints(image_path)
#             # Check if keypoints are extracted successfully
#             if len(keypoints) > 0:
#                 X_data.append(keypoints.flatten())
#                 y_data.append(folder)  # Use folder name as label
#                 count += 1  # Increment the count
#
# # Convert data lists to numpy arrays
# X = np.array(X_data)
# y = np.array(y_data)
#
# # Create and train a Multi-layer Perceptron (MLP) classifier
# model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
# model.fit(X, y)
#
# # Now the model has learned from the hand pose keypoints and their corresponding labels
# from joblib import dump
#
# # Save the trained model to an h5 file
# model_filename = "hand_pose_classifier_model.h5"
# dump(model, model_filename)
# print(f"Model saved as {model_filename}")


import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.neural_network import MLPClassifier

# Function to extract connected keypoints from an image
def extract_connected_keypoints(image_path):
    image_with_landmarks = cv2.imread(image_path)
    image_with_landmarks_rgb = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    results = hands.process(image_with_landmarks_rgb)
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
    connected_keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for connection in connections:
                kp1_index, kp2_index = connection
                kp1 = hand_landmarks.landmark[kp1_index]
                kp2 = hand_landmarks.landmark[kp2_index]
                connected_keypoints.append([kp1.x, kp1.y, kp1.z])
                connected_keypoints.append([kp2.x, kp2.y, kp2.z])
    return np.array(connected_keypoints)

# Path to the dataset folder
dataset_folder = "C:/Users/dilip/PycharmProjects/final-code-project/GCN+CNN/Dataset"

# Initialize lists to store data
X_data = []
y_data = []

# Iterate through each folder in the dataset
for folder in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder)
    if os.path.isdir(folder_path):
        count = 0
        for filename in os.listdir(folder_path):
            if count >= 10:
                break
            image_path = os.path.join(folder_path, filename)
            # Extract connected keypoints and append to the data lists
            connected_keypoints = extract_connected_keypoints(image_path)
            if len(connected_keypoints) > 0:
                X_data.append(connected_keypoints.flatten())
                y_data.append(folder)
                count += 1

# Convert data lists to numpy arrays
X = np.array(X_data)
y = np.array(y_data)

# Create and train a Multi-layer Perceptron (MLP) classifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
model.fit(X, y)

# Save the trained model to a file
from joblib import dump
model_filename = "hand_pose_classifier_model1.joblib"
dump(model, model_filename)
print(f"Model saved as {model_filename}")


# Access the learned parameters of the MLP classifier
# Weights of each layer
weights = model.coefs_
# Biases of each layer
biases = model.intercepts_

# Output the learned parameters
for i, (w, b) in enumerate(zip(weights, biases)):
    print(f"Layer {i}:")
    print("Weights shape:", w.shape)
    print("Biases shape:", b.shape)
    print("Weights:", w)
    print("Biases:", b)
