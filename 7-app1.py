# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import tkinter as tk
# from PIL import Image, ImageTk
#
# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Load the CNN model
# cnn_model = load_model('C:/Users/dilip/PycharmProjects/final-code-project/GCN+CNN/hand_sign_modelCNN.h5')
#
# # Load the MLP classifier model
# mlp_model = joblib.load("C:/Users/dilip/PycharmProjects/final-code-project/GCN+CNN/hand_pose_classifier_model.joblib")
#
# # Create a function to preprocess the input image for the CNN model
# def preprocess_image(image):
#     image = tf.image.resize(image, (224, 224))
#     image = tf.expand_dims(image, axis=0)
#     image /= 255.0  # Normalize pixel values
#     return image
#
# # Function to draw landmarks on the image
# def draw_landmarks(image, landmarks):
#     if landmarks:
#         # Draw landmarks and connect them with lines
#         for landmark in landmarks:
#             hand_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in landmark.landmark]
#             for i in range(len(hand_points)-1):
#                 cv2.line(image, hand_points[i], hand_points[i+1], (0, 255, 0), 2)  # Green color for lines
#
#             for point in hand_points:
#                 cv2.circle(image, point, 6, (0, 255, 255), -1)  # Yellow color for landmarks
#     return image
#
# # Function to extract connected keypoints from a frame with landmarks
# def extract_connected_keypoints(frame_with_landmarks, landmarks):
#     # Define connections between landmarks
#     connections = [
#         # Thumb
#         (0, 1), (1, 2), (2, 3), (3, 4),
#         # Index finger
#         (0, 5), (5, 6), (6, 7), (7, 8),
#         # Middle finger
#         (0, 9), (9, 10), (10, 11), (11, 12),
#         # Ring finger
#         (0, 13), (13, 14), (14, 15), (15, 16),
#         # Little finger
#         (0, 17), (17, 18), (18, 19), (19, 20),
#         # Connections between fingers
#         (1, 5), (5, 9), (9, 13), (13, 17)
#     ]
#
#     # Extract landmarks if hand is detected
#     connected_keypoints = []
#     if landmarks:
#         for hand_landmarks in landmarks:
#             for connection in connections:
#                 kp1_index, kp2_index = connection
#                 kp1 = hand_landmarks.landmark[kp1_index]
#                 kp2 = hand_landmarks.landmark[kp2_index]
#                 # Extract x, y, and z coordinates of keypoints
#                 connected_keypoints.extend([kp1.x, kp1.y, kp1.z, kp2.x, kp2.y, kp2.z])
#
#     return connected_keypoints
#
# # Function to get predictions from both models and choose the best one
# # Function to get predictions from both models and choose the best one
# def get_combined_prediction(image, landmarks):
#     # Preprocess the input image for the CNN model
#     cnn_input = preprocess_image(image)
#
#     # Get prediction from the CNN model
#     cnn_prediction = cnn_model.predict(cnn_input)
#     cnn_confidence = cnn_prediction[0][0]  # Extracting the confidence score from the array
#
#     # Get prediction from the MLP classifier
#     # Extract keypoints from the new image
#     new_hand_pose_keypoints = extract_connected_keypoints(image, landmarks)
#     # Convert the keypoints to a NumPy array
#     new_hand_pose_keypoints_np = np.array(new_hand_pose_keypoints)
#     # Flatten the array of keypoints and reshape it to match the input shape used during training
#     # new_flattened_keypoints = new_hand_pose_keypoints_np.flatten().reshape(1, -1)
#     # Flatten the array of keypoints
#     new_flattened_keypoints = new_hand_pose_keypoints_np.flatten()
#
#     # Ensure that the flattened array has the correct number of features (144)
#     if new_flattened_keypoints.shape[0] != 144:
#         print("Incorrect number of features. Please check the shape of the flattened array.")
#         return
#     else:
#         # Reshape the flattened array to match the input shape expected by the MLP classifier
#         new_flattened_keypoints = new_flattened_keypoints.reshape(1, -1)
#
#     # Use the trained model to predict the label
#     mlp_prediction = mlp_model.predict_proba(new_flattened_keypoints)
#     mlp_confidence = np.max(mlp_prediction)  # Extracting the maximum confidence score
#
#     # Combine predictions using a weighted average
#     combined_prediction = (cnn_confidence * cnn_prediction) + (mlp_confidence * mlp_prediction)
#     return combined_prediction
#
# # Function to capture and process the video feed
# # Function to capture and process the video feed
# def capture_video(video_panel):
#     # Capture real-time data from webcam
#     cap = cv2.VideoCapture(0)
#
#     image_saved = False
#
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#
#         # Convert the frame to RGB format
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # Resize the frame
#         frame_resized = cv2.resize(frame_rgb, (600, 600))
#
#         # Convert the frame to ImageTk format
#         img = Image.fromarray(frame_resized)
#         imgtk = ImageTk.PhotoImage(image=img)
#
#         # Update the video panel with the new frame
#         video_panel.imgtk = imgtk
#         video_panel.config(image=imgtk)
#
#         # Detect hand landmarks
#         results = hands.process(frame_rgb)
#         landmarks = results.multi_hand_landmarks
#
#         # Draw landmarks on the frame
#         frame_with_landmarks = draw_landmarks(frame.copy(), landmarks)
#
#         # Check if hands are detected
#         if landmarks:
#             for hand_landmarks in landmarks:
#                 # Extract bounding box coordinates of the detected hand
#                 hand_bbox = cv2.boundingRect(np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).astype(np.float32))
#
#                 # Draw bounding box around the detected hand
#                 cv2.rectangle(frame_with_landmarks, (hand_bbox[0], hand_bbox[1]), (hand_bbox[0] + hand_bbox[2], hand_bbox[1] + hand_bbox[3]), (0, 255, 0), 2)
#
#         # Display the captured frame with landmarks
#         cv2.imshow('Hand Recognition', frame_with_landmarks)
#
#         # Check if 's' key is pressed to save the image
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('s'):
#             # Check if landmarks are detected
#             if landmarks:
#                 # Save the image only if landmarks are present
#                 cv2.imwrite("captured_images/hand_image.jpg", frame_with_landmarks)
#                 image_saved = True
#                 print("Image saved!")
#             else:
#                 print("No landmarks detected in the video screen. Please recapture the image after 5 seconds.")
#                 time.sleep(5)
#
#         # Check if 'q' key is pressed to quit
#         if key & 0xFF == ord('q'):
#             break
#
#         # Delay before predicting keypoints
#         if image_saved:
#             # Read the saved image
#             saved_image = cv2.imread("captured_images/hand_image.jpg")
#
#             # # Extract connected keypoints from the saved image
#             connected_keypoints = extract_connected_keypoints(saved_image,landmarks)
#
#             # Predict the label only if connected keypoints are present
#             if connected_keypoints:
#                 combined_prediction = get_combined_prediction(saved_image, landmarks)
#                 print("Combined model Prediction:", combined_prediction)
#
#                 # Extract the index of the maximum confidence score
#                 predicted_label_index = np.argmax(combined_prediction)
#
#                 # Map the index to the corresponding hand sign label
#                 predicted_hand_sign = {0: "Five", 1: "Four", 2: "One", 3: "Three", 4: "Two"}
#
#                 # Get the predicted hand sign label
#                 predicted_label = predicted_hand_sign[predicted_label_index]
#
#                 print("Predicted Hand Sign:", predicted_label)
#
#                 # Reset the image saved flag
#                 image_saved = False
#
#     # Release the capture
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # Function to create a Tkinter window
# def create_gui():
#     # Create Tkinter window
#     root = tk.Tk()
#     root.title("Hand Gesture Recognition")
#
#     # Create a label to display the video
#     video_label = tk.Label(root)
#     video_label.pack()
#     # Create frame for video display
#     video_frame = tk.Frame(root, width=600, height=600, bg="white")  # Set background color to white
#     video_frame.pack(side=tk.LEFT, padx=10)
#
#     # Placeholder image for video display
#     placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (600, 600), "#353131"))
#     video_panel = tk.Label(video_frame, image=placeholder_img)
#     video_panel.pack()
#
#     # Create buttons for capturing image and quitting
#     capture_button = tk.Button(root, text="Capture Image", command=lambda: capture_video(video_panel))  # Pass video_panel as an argument
#     capture_button.pack(side=tk.LEFT)
#
#     quit_button = tk.Button(root, text="Quit", command=root.quit)
#     quit_button.pack(side=tk.RIGHT)
#
#     # Run the Tkinter event loop
#     root.mainloop()
#
# # Call the function to create the GUI
# create_gui()

import keyboard
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
from tkinterweb import HtmlFrame  # Import the HTML browser
import pyautogui
import autopy
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math


# Initialize the volume object
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Define minVol and maxVol
minVol = -63
maxVol = 0

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the CNN model
cnn_model = load_model('C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_sign_model4.h5')

# Load the MLP classifier model
mlp_model = joblib.load("C:/Users/dilip/PycharmProjects/Final-Year-Project/hand_pose_classifier_model1.joblib")

# Create a function to preprocess the input image for the CNN model
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    image /= 255.0  # Normalize pixel values
    return image

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
def extract_connected_keypoints(frame_with_landmarks, landmarks):
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
                # Extract x, y, and z coordinates of keypoints
                connected_keypoints.extend([kp1.x, kp1.y, kp1.z, kp2.x, kp2.y, kp2.z])

    return connected_keypoints

# Function to get predictions from both models and choose the best one
def get_combined_prediction(image, landmarks):
    # Preprocess the input image for the CNN model
    cnn_input = preprocess_image(image)

    # Get prediction from the CNN model
    cnn_prediction = cnn_model.predict(cnn_input)
    cnn_confidence = cnn_prediction[0][0]  # Extracting the confidence score from the array

    # Get prediction from the MLP classifier
    # Extract keypoints from the new image
    new_hand_pose_keypoints = extract_connected_keypoints(image, landmarks)
    # Convert the keypoints to a NumPy array
    new_hand_pose_keypoints_np = np.array(new_hand_pose_keypoints)
    # Flatten the array of keypoints and reshape it to match the input shape used during training
    # new_flattened_keypoints = new_hand_pose_keypoints_np.flatten().reshape(1, -1)
    # Flatten the array of keypoints
    new_flattened_keypoints = new_hand_pose_keypoints_np.flatten()

    # Ensure that the flattened array has the correct number of features (144)
    if new_flattened_keypoints.shape[0] != 144:
        print("Incorrect number of features. Please check the shape of the flattened array.")
        return
    else:
        # Reshape the flattened array to match the input shape expected by the MLP classifier
        new_flattened_keypoints = new_flattened_keypoints.reshape(1, -1)

    # Use the trained model to predict the label
    mlp_prediction = mlp_model.predict_proba(new_flattened_keypoints)
    mlp_confidence = np.max(mlp_prediction)  # Extracting the maximum confidence score

    # Combine predictions using a weighted average
    combined_prediction = (cnn_confidence * cnn_prediction) + (mlp_confidence * mlp_prediction)
    return combined_prediction

# Function to capture and process the video feed
def capture_video():
    # Capture real-time data from webcam
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    image_saved = False
    frames = []

    tipIds = [4, 8, 12, 16, 20]
    mode = ''
    active = 0

    pyautogui.FAILSAFE = False

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
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                # Draw bounding box around the detected hand
                cv2.rectangle(frame_with_landmarks, (hand_bbox[0], hand_bbox[1]), (hand_bbox[0] + hand_bbox[2], hand_bbox[1] + hand_bbox[3]), (0, 255, 0), 2)

                fingers = []

            if len(lmList) != 0:
                # Thumb
                if lmList[tipIds[0]][1] > lmList[tipIds[0 - 1]][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                if (fingers == [0, 0, 0, 0, 0]) and (active == 0):
                    mode = 'N'
                elif ((fingers == [0, 1, 1, 1, 1]) or (fingers == [0, 1, 1, 1, 1])) and (active == 0):
                    mode = 'Scroll'
                    active = 1
                elif (fingers == [1, 1, 0, 0, 0]) and (active == 0):
                    mode = 'Volume'
                    active = 1
                elif (fingers == [1, 1, 1, 1, 1]) and (active == 0):
                    mode = 'Cursor'
                    active = 1

                if mode == 'Scroll':
                    active = 1
                    cv2.rectangle(frame, (200, 410), (245, 460), (255, 255, 255), cv2.FILLED)
                    if fingers == [0, 1, 1, 1, 1]:
                        pyautogui.scroll(300)  # Scroll up
                    elif fingers == [0, 1, 1, 1, 0]:  # Adjusted finger configuration for scrolling down
                        pyautogui.scroll(-300)  # Scroll down
                    elif fingers == [0, 0, 0, 0, 0]:
                        active = 0
                        mode = 'N'


                elif mode == 'Volume':
                    active = 1
                    if fingers[-1] == 1:
                        active = 0
                        mode = 'N'
                    else:
                        x1, y1 = lmList[4][1], lmList[4][2]
                        x2, y2 = lmList[8][1], lmList[8][2]
                        length = math.hypot(x2 - x1, y2 - y1)
                        vol = np.interp(length, [50, 200], [minVol, maxVol])
                        volN = int(vol)
                        if volN % 4 != 0:
                            volN = volN - volN % 4
                            if volN >= 0:
                                volN = 0
                            elif volN <= -64:
                                volN = -64
                            elif vol >= -11:
                                volN = vol

                        volume.SetMasterVolumeLevel(vol, None)

                elif mode == 'Cursor':
                    active = 1
                    cv2.rectangle(frame, (110, 20), (620, 350), (255, 255, 255), 3)
                    if fingers[1:] == [0, 0, 0, 0]:
                        active = 0
                        mode = 'N'
                    else:
                        x1, y1 = lmList[8][1], lmList[8][2]
                        w, h = autopy.screen.size()
                        X = int(np.interp(x1, [110, 620], [0, w - 1]))
                        Y = int(np.interp(y1, [20, 350], [0, h - 1]))
                        if X % 2 != 0:
                            X = X - X % 2
                        if Y % 2 != 0:
                            Y = Y - Y % 2
                        autopy.mouse.move(X, Y)
                        if fingers[0] == 0:
                            pyautogui.click()

        # Display the captured frame with landmarks
        # Resize the frame
        frame_with_landmarks = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_with_landmarks, (600, 600))

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video panel with the new frame
        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

        # Check if 's' key is pressed to save the image
        if keyboard.is_pressed('s'):
            print("S key pressed")
            # Check if landmarks are detected
            if landmarks:
                # Save the image only if landmarks are present
                cv2.imwrite("captured_images/hand_image.jpg", cv2.cvtColor(frame_with_landmarks, cv2.COLOR_RGB2BGR))  # Save the image in BGR format
                image_saved = True
                print("Image saved!")
            else:
                print("No landmarks detected in the video screen. Please recapture the image after 5 seconds.")
                time.sleep(5)

        # Check if 'q' key is pressed to quit
        if keyboard.is_pressed('q'):
            print("Q key pressed")
            break

        # Delay before predicting keypoints
        if image_saved:
            # Read the saved image
            saved_image = cv2.imread("captured_images/hand_image.jpg")

            # # Extract connected keypoints from the saved image
            connected_keypoints = extract_connected_keypoints(saved_image,landmarks)

            # Predict the label only if connected keypoints are present
            if connected_keypoints:
                combined_prediction = get_combined_prediction(saved_image, landmarks)
                print("Combined model Prediction:", combined_prediction)

                # Extract the index of the maximum confidence score
                predicted_label_index = np.argmax(combined_prediction)

                # Map the index to the corresponding hand sign label
                predicted_hand_sign = {0: "Five", 1: "Four", 2: "One", 3: "Three", 4: "Two"}

                # Get the predicted hand sign label
                predicted_label = predicted_hand_sign[predicted_label_index]

                print("Predicted Hand Sign:", predicted_label)
                # Destroy the previous frame if it exists
                for frame in frames:
                    frame.destroy()


                if predicted_label == "Five":
                    new_frame  = HtmlFrame(chat_frame, width=600, height=600)
                    new_frame .load_website("https://www.google.com")  # Load a HTTPS website
                    new_frame .pack(fill="both", expand=True)
                    frames.append(new_frame)
                elif predicted_label == "Four":
                    new_frame  = HtmlFrame(chat_frame, width=600, height=600)
                    new_frame .load_website("https://www.wikipedia.org/")  # Load a HTTPS website
                    new_frame .pack(fill="both", expand=True)
                    frames.append(new_frame)
                # Reset the image saved flag
                image_saved = False

        # Update the Tkinter GUI
        root.update()

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Function to create a Tkinter window
# Create Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.configure(bg="#303030")  # Set background color to blue
root.attributes("-fullscreen", True)  # Open the window in full screen mode

# Create frame for video display
video_frame = tk.Frame(root, width=600, height=600, bg="white")  # Set background color to white
video_frame.pack(side=tk.LEFT, padx=10)

placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (600, 600), "#1a1a1a"))
video_panel = tk.Label(video_frame, image=placeholder_img)
video_panel.pack()


# Create frame for buttons below the video display
button_frame = tk.Frame(root, bg="#303030")
button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

# Create buttons for capturing image and quitting
capture_button = tk.Button(button_frame, text="Capture Image", command=capture_video)
capture_button.pack(side=tk.LEFT, padx=10)

quit_button = tk.Button(button_frame, text="Quit", command=root.quit)
quit_button.pack(side=tk.RIGHT, padx=10)


# Create label for detected hand sign
detected_sign_label = tk.Label(root, text="", font=("Arial", 16), bg="#303030", fg="black")
detected_sign_label.pack(pady=10)

# Create frame for chatbox on the right side
chat_frame = tk.Frame(root, width=600, height=600, bg="#1a1a1a")
chat_frame.pack(side=tk.RIGHT, padx=10)

# Run the Tkinter event loop
root.mainloop()

