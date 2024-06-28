import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer,MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import datetime
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def visual_dct_load_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        for subDir in child.glob('*'):
            binary_file = None
            for subsubDir in subDir.glob('*'):
                subsubDir = str(subsubDir)
                if 'idct' in subsubDir.split('_'):  # Assuming 'idct' is your visual data
                    binary_file = subsubDir

            if binary_file is not None:
                visual_feat = np.load(binary_file)
                data.append(visual_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    ##### our original label encoder #####
    # le = LabelEncoder().fit(labels)
    # encoded_labels = to_categorical(le.transform(labels))
    ######################################

    ##### label encoder used in lab example #####
    LE = LabelEncoder()
    encoded_labels = to_categorical(LE.fit_transform(labels))
    ############################################

    return data, encoded_labels


def check_data_scaling(data, n_features):
    # Reshape data to 2D if it's not already, assuming the last dimension is features
    if data.ndim > 2:
        n_samples = data.shape[0] * data.shape[1]
        data = data.reshape((n_samples, n_features))

    # Calculate and print descriptive statistics
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)

    print("Means:", means)
    print("Standard Deviations:", stds)
    print("Min:", mins)
    print("Max:", maxs)

    # Number of features to plot per figure
    features_per_figure = 5

    # Plot histograms for each feature across all samples
    for start_idx in range(0, n_features, features_per_figure):
        end_idx = start_idx + features_per_figure
        fig, axs = plt.subplots(features_per_figure, 1, figsize=(10, 5 * features_per_figure))

        # If there's less than `features_per_figure` at the end, reduce the number of subplots
        if end_idx > n_features:
            end_idx = n_features
            rows_to_plot = end_idx - start_idx
            fig, axs = plt.subplots(rows_to_plot, 1, figsize=(10, 5 * rows_to_plot))

        for i in range(start_idx, end_idx):
            ax_idx = i - start_idx
            axs[ax_idx].hist(data[:, i], bins=50, color='blue', edgecolor='black')
            axs[ax_idx].set_title(f'Feature {i}')

        plt.tight_layout()
        plt.show()

def main():
    # Load the visual data and labels
    visual_dct_features, labels = visual_dct_load_data()

    # Determine the shape of the visual features
    n_samples, n_time_steps, n_features = visual_dct_features.shape

    # Check if the data is scaled or normalized before scaling
    check_data_scaling(visual_dct_features, n_features)

    # Flatten the 3D feature array to 2D for scaling
    visual_dct_features_2d = visual_dct_features.reshape((n_samples * n_time_steps, n_features))

    # Initialize the standard scaler
    scaler = StandardScaler()

    # Fit the scaler to the flattened data and transform it
    visual_dct_features_scaled_2d = scaler.fit_transform(visual_dct_features_2d)

    # Reshape the scaled features back to 3D
    visual_dct_features_scaled = visual_dct_features_scaled_2d.reshape((n_samples, n_time_steps, n_features))

    # Optionally, check scaling after standardization
    check_data_scaling(visual_dct_features_scaled, n_features)

if __name__ == "__main__":
    main()