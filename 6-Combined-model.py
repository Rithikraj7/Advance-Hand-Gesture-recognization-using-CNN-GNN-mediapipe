import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import joblib
import numpy as np
import mediapipe as mp
# Load the CNN model
cnn_model = load_model('C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_sign_model4.h5')

# Load the MLP classifier model
mlp_model = joblib.load("C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_pose_classifier_model1.joblib")

# Create a function to preprocess the input image for the CNN model
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image /= 255.0  # Normalize pixel values
    return image


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
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Little finger
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Connections between fingers
        (1, 5), (5, 9), (9, 13), (13, 17)
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


# Function to get predictions from both models and choose the best one
# Function to get predictions from both models and choose the best one
def get_combined_prediction(image_path):
    # Preprocess the input image for the CNN model
    cnn_input = preprocess_image(image_path)

    # Get prediction from the CNN model
    cnn_prediction = cnn_model.predict(cnn_input)
    print("cnn_prediction :",cnn_prediction)
    cnn_confidence = cnn_prediction[0][0]  # Extracting the confidence score from the array

    # Get prediction from the MLP classifier
    # Extract keypoints from the new image
    new_hand_pose_keypoints = extract_connected_keypoints(image_path)
    # Convert the keypoints to a NumPy array
    new_hand_pose_keypoints_np = np.array(new_hand_pose_keypoints)
    # Flatten the array of keypoints and reshape it to match the input shape used during training
    new_flattened_keypoints = new_hand_pose_keypoints_np.flatten().reshape(1, -1)
    # Use the trained model to predict the label
    mlp_prediction = mlp_model.predict_proba(new_flattened_keypoints)
    print("mlp_prediction :",mlp_prediction)
    mlp_confidence = np.max(mlp_prediction)  # Extracting the maximum confidence score

    # Combine predictions using a weighted average
    combined_prediction = (cnn_confidence * cnn_prediction) + (mlp_confidence * mlp_prediction)
    return combined_prediction

Label = ["Five","Four","One","Three","Two"]
print("Label: ",Label)
# Test the combined model with an example image
image_path = "C:/Users/dilip/PycharmProjects/Final-Year-Project/prediction/Four_2.jpg"
combined_prediction = get_combined_prediction(image_path)
print("Combined model Prediction:", combined_prediction)

# Extract the index of the maximum confidence score
predicted_label_index = np.argmax(combined_prediction)

# Map the index to the corresponding hand sign label
predicted_hand_sign = {0: "Five", 1: "Four", 2: "One", 3: "Three", 4: "Two"}

# Get the predicted hand sign label
predicted_label = predicted_hand_sign[predicted_label_index]

print("Predicted Hand Sign:", predicted_label)

# Get the ground truth label (assuming you have it)
ground_truth_label = "Four"  # Replace this with the actual ground truth label

# Check if the predicted label matches the ground truth label
if predicted_label == ground_truth_label:
    accuracy = 1.0  # If they match, accuracy is 100%
else:
    accuracy = 0.0  # If they don't match, accuracy is 0%

print("Accuracy:", accuracy)
