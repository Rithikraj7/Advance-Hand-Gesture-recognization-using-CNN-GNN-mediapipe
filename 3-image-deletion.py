import os

def delete_non_matching_files(data_folder, adjacency_folder):
    # Iterate over each subfolder in the data folder
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)

        # Check if the subfolder is a directory
        if os.path.isdir(label_folder):
            # Iterate over each image file in the subfolder
            for filename in os.listdir(label_folder):
                # Check if the file is an image
                if filename.endswith(".jpg"):
                    # Construct the filename for the adjacency matrix file
                    adjacency_filename = f"adjacency_matrix_{filename[:-4]}.txt"
                    adjacency_path = os.path.join(adjacency_folder, label, adjacency_filename)

                    # Check if the adjacency matrix file exists
                    if not os.path.exists(adjacency_path):
                        # Print a message indicating the non-matching file
                        print(f"Deleting non-matching image: {filename}")
                        # Delete the image file
                        os.remove(os.path.join(label_folder, filename))

# Specify the paths to the data folder and adjacency folder
data_folder = "Dataset/"
adjacency_folder = "LANDMARKS/"

# Call the function to delete non-matching files
delete_non_matching_files(data_folder, adjacency_folder)
