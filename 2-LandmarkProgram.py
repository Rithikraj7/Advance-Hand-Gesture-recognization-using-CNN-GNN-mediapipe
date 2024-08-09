# the code is used to get the landmarks of the images
# import cv2
# import mediapipe as mp
# import os
#
# def detect_and_save_landmarks(image_path, output_file):
#     # Initialize MediaPipe Hand module
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands()
#
#     # Check if the file exists
#     if not os.path.exists(image_path):
#         print(f"Error: Image file not found - {image_path}")
#         return
#
#     # Read the image
#     image = cv2.imread(image_path)
#
#     # Check if the image is successfully loaded
#     if image is not None:
#         # Convert the image to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Run hand landmark detection
#         results = hands.process(image_rgb)
#
#         # Check if Landmarks_folder are detected
#         if results.multi_hand_landmarks:
#             detected_landmarks = []
#
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Save the Landmarks_folder
#                 landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
#                 detected_landmarks.extend(landmarks)
#
#             # Save Landmarks_folder to a file
#             with open(output_file, 'w') as file:
#                 for landmark in detected_landmarks:
#                     file.write(f"{landmark[0]},{landmark[1]},{landmark[2]}\n")
#
#             print(f"Landmarks saved to {output_file}")
#
#         else:
#             print(f"No hand Landmarks_folder detected in the image: {image_path}")
#     else:
#         print(f"Error: Unable to read the image from {image_path}")
#
# if __name__ == "__main__":
#     # Path to your dataset images
#     dataset_path = "C:/Users/dilip/PycharmProjects/pythonfinalpg/NewDataset/Two/"
#
#     # Output directory for saving landmark files
#     output_dir = "Landmarks_folder/"
#
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Loop through images in the dataset and detect Landmarks_folder
#     for i in range(1, 101):  # Assuming you have 100 images in the dataset
#         image_path = f"{dataset_path}Two_{i}.jpg"  # Corrected file name
#         output_file = f"{output_dir}Two_{i}_landmarks.txt"
#         detect_and_save_landmarks(image_path, output_file)


# this code is used to get the adjucency matrix from the landmakr files
# import os
# import numpy as np
#
# # Function to load landmark coordinates from file
# def load_landmark_coordinates(file_path):
#     with open(file_path, 'r') as file:
#         landmark_coordinates = [list(map(float, line.strip().split(','))) for line in file]
#     return landmark_coordinates
#
# # Function to calculate distances between landmarks
# def calculate_distances(landmark_coordinates):
#     num_landmarks = len(landmark_coordinates)
#     distances = np.zeros((num_landmarks, num_landmarks))
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):
#             distance = np.linalg.norm(np.array(landmark_coordinates[i]) - np.array(landmark_coordinates[j]))
#             distances[i, j] = distance
#             distances[j, i] = distance  # Symmetric matrix
#     return distances
#
# # Function to generate adjacency matrix based on distances and threshold
# def generate_adjacency_matrix(distances, threshold):
#     adjacency_matrix = (distances < threshold).astype(int)
#     return adjacency_matrix
#
# # Function to write adjacency matrix to file
# def write_adjacency_matrix_to_file(adjacency_matrix, file_path):
#     with open(file_path, 'w') as file:
#         for row in adjacency_matrix:
#             file.write(' '.join(map(str, row)) + '\n')
#
# # Main function to generate adjacency matrix text files
# def generate_adjacency_matrix_files(landmarks_folder, output_folder, threshold=0.2):
#     # Iterate over each class folder
#     for class_folder in os.listdir(landmarks_folder):
#         class_path = os.path.join(landmarks_folder, class_folder)
#         if os.path.isdir(class_path):
#             output_class_folder = os.path.join(output_folder, class_folder)
#             os.makedirs(output_class_folder, exist_ok=True)
#             for landmark_file in os.listdir(class_path):
#                 if landmark_file.endswith('_landmarks.txt'):
#                     # Load landmark coordinates
#                     landmark_file_path = os.path.join(class_path, landmark_file)
#                     landmark_coordinates = load_landmark_coordinates(landmark_file_path)
#                     # Calculate distances
#                     distances = calculate_distances(landmark_coordinates)
#                     # Generate adjacency matrix
#                     adjacency_matrix = generate_adjacency_matrix(distances, threshold)
#                     # Write adjacency matrix to file
#                     output_file_name = f'adjacency_matrix_{landmark_file[:-len("_landmarks.txt")]}.txt'
#                     output_file_path = os.path.join(output_class_folder, output_file_name)
#                     write_adjacency_matrix_to_file(adjacency_matrix, output_file_path)
#
# # Example usage
# landmarks_folder = 'Landmarks_folder/'
# output_folder = 'Adjacency_matrices'
# generate_adjacency_matrix_files(landmarks_folder, output_folder, threshold=0.2)

# import cv2
# import mediapipe as mp
# import os
#
# def detect_and_save_landmarks(image, output_file):
#     # Initialize MediaPipe Hand module
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands()
#
#     # Convert the image to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Run hand landmark detection
#     results = hands.process(image_rgb)
#
#     # Check if landmarks are detected
#     if results.multi_hand_landmarks:
#         detected_landmarks = []
#
#         # Counter for landmarks
#         landmark_count = 0
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Save the landmarks
#             landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
#
#             # Save only the first 10 landmarks
#             for landmark in landmarks:
#                 detected_landmarks.append(landmark)
#                 landmark_count += 1
#
#                 if landmark_count >= 10:
#                     break
#
#             if landmark_count >= 10:
#                 break
#
#         # Save landmarks to a file
#         with open(output_file, 'w') as file:
#             for landmark in detected_landmarks:
#                 file.write(f"{landmark[0]},{landmark[1]},{landmark[2]}\n")
#
#         print(f"Landmarks saved to {output_file}")
#
#         return True  # Return True if landmarks are saved
#     else:
#         print(f"No hand landmarks detected in the image.")
#         return False  # Return False if no landmarks are detected
#
# def check_landmarks_count(class_output_dir):
#     # Check the number of files in the class output directory
#     file_count = len([name for name in os.listdir(class_output_dir) if os.path.isfile(os.path.join(class_output_dir, name))])
#     return file_count
#
# if __name__ == "__main__":
#     # Output directory for saving landmark files
#     output_dir = "Landmarks_folder1/"
#
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Loop through classes in the dataset
#     classes = ["Five", "Four", "One", "Three", "Two"]
#
#     for class_name in classes:
#         print(f"Processing class: {class_name}")
#         class_images_path = f"C:/Users/dilip/PycharmProjects/pythonfinalpg/NewDataset/{class_name}/"
#
#         # Create the class output directory if it doesn't exist
#         class_output_dir = os.path.join(output_dir, class_name)
#         os.makedirs(class_output_dir, exist_ok=True)
#
#         # Check if the class already has 10 landmarks
#         if check_landmarks_count(class_output_dir) >= 10:
#             print(f"Class {class_name} already has 10 landmarks. Skipping...")
#             continue
#
#         # Loop through images in the class
#         for image_name in os.listdir(class_images_path):
#             image_path = os.path.join(class_images_path, image_name)
#
#             # Read the image
#             image = cv2.imread(image_path)
#
#             # Check if the image is successfully loaded
#             if image is not None:
#                 # Define the output file path
#                 output_file = os.path.join(class_output_dir, f"{os.path.splitext(image_name)[0]}_landmarks.txt")
#
#                 # Detect and save landmarks
#                 if detect_and_save_landmarks(image, output_file):
#                     # Check if we have saved 10 landmarks for the class
#                     if check_landmarks_count(class_output_dir) >= 10:
#                         print(f"Class {class_name} now has 10 landmarks. Moving to the next class...")
#                         break
#             else:
#                 print(f"Error: Unable to read the image from {image_path}")


#============working code for generating the and detecting the land mark =======================
# import cv2
# import mediapipe as mp
# import os
# import threading
#
# # Initialize MediaPipe Hand module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# def detect_and_save_landmarks(image, output_file):
#     # Convert the image to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Run hand landmark detection
#     results = hands.process(image_rgb)
#
#     # Check if landmarks are detected
#     if results.multi_hand_landmarks:
#         detected_landmarks = []
#
#         # Counter for landmarks
#         landmark_count = 0
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Save the landmarks
#             landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
#
#             # Save only the first 10 landmarks
#             for landmark in landmarks:
#                 detected_landmarks.append(landmark)
#                 landmark_count += 1
#
#                 if landmark_count >= 10:
#                     break
#
#             if landmark_count >= 10:
#                 break
#
#         # Save landmarks to a file
#         with open(output_file, 'w') as file:
#             for landmark in detected_landmarks:
#                 file.write(f"{landmark[0]},{landmark[1]},{landmark[2]}\n")
#
#         print(f"Landmarks saved to {output_file}")
#
#         return True  # Return True if landmarks are saved
#     else:
#         print(f"No hand landmarks detected in the image.")
#         return False  # Return False if no landmarks are detected
#
# def process_images_in_class(class_name, class_images_path, class_output_dir):
#     print(f"Processing class: {class_name}")
#
#     # Loop through images in the class
#     for image_name in os.listdir(class_images_path):
#         image_path = os.path.join(class_images_path, image_name)
#
#         # Read the image
#         image = cv2.imread(image_path)
#
#         # Check if the image is successfully loaded
#         if image is not None:
#             # Define the output file path
#             output_file = os.path.join(class_output_dir, f"{os.path.splitext(image_name)[0]}_landmarks.txt")
#
#             # Detect and save landmarks
#             detect_and_save_landmarks(image, output_file)
#         else:
#             print(f"Error: Unable to read the image from {image_path}")
#
# if __name__ == "__main__":
#     # Output directory for saving landmark files
#     output_dir = "Landmarks_folder1/"
#
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Loop through classes in the dataset
#     classes = ["Five", "Four", "One", "Three", "Two"]
#
#     # Create and start a thread for each class
#     threads = []
#     for class_name in classes:
#         class_images_path = f"Dataset/{class_name}/"
#         class_output_dir = os.path.join(output_dir, class_name)
#         os.makedirs(class_output_dir, exist_ok=True)
#
#         thread = threading.Thread(target=process_images_in_class, args=(class_name, class_images_path, class_output_dir))
#         threads.append(thread)
#         thread.start()
#
#     # Wait for all threads to finish
#     for thread in threads:
#         thread.join()
#
#     # Close the MediaPipe hands instance
#     hands.close()
#

import cv2
import mediapipe as mp
import os
import threading

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def detect_and_save_landmarks(image, output_file):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run hand landmark detection
    results = hands.process(image_rgb)

    # Check if landmarks are detected
    if results.multi_hand_landmarks:
        detected_landmarks = []

        # Counter for landmarks
        landmark_count = 0

        for hand_landmarks in results.multi_hand_landmarks:
            # Save the landmarks
            landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]

            # Save only the first 10 landmarks
            for landmark in landmarks:
                detected_landmarks.append(landmark)
                landmark_count += 1

                if landmark_count >= 10:
                    break

            if landmark_count >= 10:
                break

        # Save landmarks to a file
        with open(output_file, 'w') as file:
            for landmark in detected_landmarks:
                file.write(f"{landmark[0]},{landmark[1]},{landmark[2]}\n")

        print(f"Landmarks saved to {output_file}")

        return True  # Return True if landmarks are saved
    else:
        print(f"No hand landmarks detected in the image.")
        return False  # Return False if no landmarks are detected

def process_images_in_class(class_name, class_images_path, class_output_dir):
    print(f"Processing class: {class_name}")

    # Loop through images in the class
    for image_name in os.listdir(class_images_path):
        image_path = os.path.join(class_images_path, image_name)

        # Read the image
        image = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if image is not None:
            # Define the output file path
            output_file = os.path.join(class_output_dir, f"{os.path.splitext(image_name)[0]}_landmarks.txt")

            # Detect and save landmarks
            detect_and_save_landmarks(image, output_file)
        else:
            print(f"Error: Unable to read the image from {image_path}")

if __name__ == "__main__":
    # Output directory for saving landmark files
    output_dir = "Landmarks_folder1/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through classes in the dataset
    classes = ["Five", "Four", "One", "Three", "Two"]

    # Process images in each class sequentially
    for class_name in classes:
        class_images_path = f"Dataset1/{class_name}/"
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Process images in the class
        process_images_in_class(class_name, class_images_path, class_output_dir)

    # Close the MediaPipe hands instance
    hands.close()
