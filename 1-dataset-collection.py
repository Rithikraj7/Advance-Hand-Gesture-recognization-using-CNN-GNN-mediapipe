# import cv2
# import mediapipe as mp
# import os
# import numpy as np
# import time
#
# # Create a directory to store the dataset
# dataset_dir = "Dataset/"
# if not os.path.exists(dataset_dir):
#     os.makedirs(dataset_dir)
#
# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Video capture
# cap = cv2.VideoCapture(0)
#
# # Map for hand signs and corresponding labels
# hand_signs = {
#     1: "One",
#     2: "Two",
#     3: "Three",
#     4: "Four",
#     5: "Five"
# }
#
# # Function to draw landmarks on the image
# def draw_landmarks(image, landmarks):
#     if landmarks:
#         # Draw landmarks and connect them with lines
#         for landmark in landmarks:
#             hand_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in landmark.landmark]
#             for i in range(len(hand_points)-1):
#                 cv2.circle(image, hand_points[i], 6, (0, 255, 255), -1)  # Yellow color for landmarks
#                 cv2.line(image, hand_points[i], hand_points[i+1], (0, 255, 255), 2)
#
#             cv2.circle(image, hand_points[-1], 6, (0, 255, 255), -1)  # Yellow color for landmarks
#     return image
#
# for sign_id, sign_name in hand_signs.items():
#     print(f"Collecting data for {sign_name}...")
#
#     sign_dir = os.path.join(dataset_dir, sign_name)
#     if not os.path.exists(sign_dir):
#         os.makedirs(sign_dir)
#
#     img_count = 0
#
#     while img_count < 100:
#         _, frame = cap.read()
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Detect hand landmarks
#         results = hands.process(frame_rgb)
#         landmarks = results.multi_hand_landmarks
#
#         # Draw landmarks on the frame
#         frame_with_landmarks = draw_landmarks(frame.copy(), landmarks)
#
#         if landmarks:
#             # Save the hand sign image
#             img_count += 1
#             sign_image = cv2.resize(frame_with_landmarks, (224, 224))
#             file_name_path = os.path.join(sign_dir, f"{sign_name}_{img_count}.jpg")
#             cv2.imwrite(file_name_path, sign_image)
#
#             cv2.putText(frame_with_landmarks, f"Collecting {img_count}/100 for {sign_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 255, 255))  # Yellow color for text
#
#         cv2.imshow('Hand Sign Capture', frame_with_landmarks)
#         k = cv2.waitKey(10)
#         if k == 27:
#             break
#
#     # Add a delay of 5 seconds before collecting data for the next label
#     time.sleep(5)
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Create a directory to store the dataset
dataset_dir = "Dataset1/"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)

# Map for hand signs and corresponding labels
hand_signs = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five"
}

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

for sign_id, sign_name in hand_signs.items():
    print(f"Collecting data for {sign_name}...")

    sign_dir = os.path.join(dataset_dir, sign_name)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    img_count = 0

    while img_count < 100:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(frame_rgb)
        landmarks = results.multi_hand_landmarks

        # Draw landmarks on the frame
        frame_with_landmarks = draw_landmarks(frame.copy(), landmarks)

        if landmarks:
            # Save the hand sign image
            img_count += 1
            sign_image = cv2.resize(frame_with_landmarks, (224, 224))
            file_name_path = os.path.join(sign_dir, f"{sign_name}_{img_count}.jpg")
            cv2.imwrite(file_name_path, sign_image)

            cv2.putText(frame_with_landmarks, f"Collecting {img_count}/100 for {sign_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255))  # Yellow color for text

        cv2.imshow('Hand Sign Capture', frame_with_landmarks)
        k = cv2.waitKey(10)
        if k == 27:
            break

    # Add a del
    # ay of 5 seconds before collecting data for the next label
    time.sleep(5)

cap.release()
cv2.destroyAllWindows()
