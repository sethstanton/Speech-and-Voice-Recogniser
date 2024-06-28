import os
import numpy as np
import fnmatch

def find_highest_shape(directory, pattern):
    highest_shape = (0, 0)  # Assuming 2D shape, adjust if necessary

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            file_path = os.path.join(root, file)
            try:
                data = np.load(file_path)
                if data.shape > highest_shape:
                    highest_shape = data.shape
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return highest_shape

# Set the directory path
directory_path = "recordings/Bonney"
pattern = '*_mfcc_features.npy'  # Pattern to match the filenames
highest_shape = find_highest_shape(directory_path, pattern)
print("Highest shape found:", highest_shape)