import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cnn_model = load_model('C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_sign_model4.h5')

# Create a function to preprocess the input image for the CNN model
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    image /= 255.0  # Normalize pixel values
    return image

mlp_model = joblib.load("C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_pose_classifier_model1.joblib")

# Function to draw landmarks on the image
def draw_landmarks(image, landmarks):
    if landmarks:
        # Draw landmarks and connect them with lines
        for landmark in landmarks:
            hand_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in landmark.landmark]
            for i in range(len(hand_points)-1):
                cv2.line(image, hand_points[i], hand_points[i+1], (0, 255, 0), 2)  # Green color for lines

            for point in hand_points:
                cv2.circle(image, point, 6, (0, 255, 255), -1)  # Yellow color for landmarks
    return image

# Function to extract connected keypoints from a frame with landmarks
def extract_connected_keypoints(frame_with_landmarks):
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
    if landmarks:
        for hand_landmarks in landmarks:
            for connection in connections:
                kp1_index, kp2_index = connection
                kp1 = hand_landmarks.landmark[kp1_index]
                kp2 = hand_landmarks.landmark[kp2_index]
                connected_keypoints.extend([kp1.x, kp1.y, kp1.z, kp2.x, kp2.y, kp2.z])

    return connected_keypoints


# Function to get predictions from both models and choose the best one
def get_combined_prediction(image):
    # Preprocess the input image for the CNN model
    cnn_input = preprocess_image(image)

    # Get prediction from the CNN model
    cnn_prediction = cnn_model.predict(cnn_input)
    cnn_confidence = cnn_prediction[0][0]  # Extracting the confidence score from the array

    # Get prediction from the MLP classifier
    # Extract keypoints from the new image
    new_hand_pose_keypoints = extract_connected_keypoints(image)
    # Convert the keypoints to a NumPy array
    new_hand_pose_keypoints_np = np.array(new_hand_pose_keypoints)
    # Flatten the array of keypoints and reshape it to match the input shape used during training
    new_flattened_keypoints = new_hand_pose_keypoints_np.flatten().reshape(1, -1)
    # Use the trained model to predict the label
    mlp_prediction = mlp_model.predict_proba(new_flattened_keypoints)
    mlp_confidence = np.max(mlp_prediction)  # Extracting the maximum confidence score

    # Combine predictions using a weighted average
    combined_prediction = (cnn_confidence * cnn_prediction) + (mlp_confidence * mlp_prediction)
    return combined_prediction


# Capture real-time data from webcam
cap = cv2.VideoCapture(0)

image_saved = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)
    landmarks = results.multi_hand_landmarks

    # Draw landmarks on the frame
    frame_with_landmarks = draw_landmarks(frame.copy(), landmarks)

    # Check if hands are detected
    if landmarks:
        for hand_landmarks in landmarks:
            # Extract bounding box coordinates of the detected hand
            hand_bbox = cv2.boundingRect(np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).astype(np.float32))

            # Draw bounding box around the detected hand
            cv2.rectangle(frame_with_landmarks, (hand_bbox[0], hand_bbox[1]), (hand_bbox[0] + hand_bbox[2], hand_bbox[1] + hand_bbox[3]), (0, 255, 0), 2)

    # Display the captured frame
    cv2.imshow('Hand Recognition', frame_with_landmarks)

    # Check if 's' key is pressed to save the image
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # Check if landmarks are detected
        if landmarks:
            # Save the image only if landmarks are present
            cv2.imwrite("captured_images/hand_image.jpg", frame_with_landmarks)
            image_saved = True
            print("Image saved!")
        else:
            print("No landmarks detected in the video screen. Please recapture the image after 5 seconds.")
            time.sleep(5)

    # Check if 'q' key is pressed to quit
    if key & 0xFF == ord('q'):
        break

    # Delay before predicting keypoints
    if image_saved:
        # Read the saved image
        saved_image = cv2.imread("captured_images/hand_image.jpg")

        # Extract connected keypoints from the saved image
        connected_keypoints = extract_connected_keypoints(saved_image)

        # Predict the label only if connected keypoints are present
        if connected_keypoints:
            combined_prediction = get_combined_prediction(saved_image)
            print("Combined model Prediction:", combined_prediction)

            # Extract the index of the maximum confidence score
            predicted_label_index = np.argmax(combined_prediction)

            # Map the index to the corresponding hand sign label
            predicted_hand_sign = {0: "Five", 1: "Four", 2: "One", 3: "Three", 4: "Two"}

            # Get the predicted hand sign label
            predicted_label = predicted_hand_sign[predicted_label_index]

            print("Predicted Hand Sign:", predicted_label)

            # Reset the image saved flag
            image_saved = False

# Release the capture
cap.release()
cv2.destroyAllWindows()
