import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer,MaxPooling2D
from keras.optimizers import Adam
import json
import datetime
from pathlib import Path

def audio_only_load_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        for subDir in child.glob('*'):
            mfcc_file = None
            for subsubDir in subDir.glob('*'):
                subsubDir = str(subsubDir)
                if 'mfcc' in subsubDir.split('_'):
                    mfcc_file = subsubDir

            if mfcc_file is not None:
                audio_feat = np.load(mfcc_file)
                data.append(audio_feat)
                label = str(subDir).split('\\')[1]  # Get label from directory name
                labels.append(label)  # Append label for each data sample

    data = np.array(data)
    labels = np.array(labels)
    print("MFCC Data shape:", data.shape)

    LE = LabelEncoder()
    encoded_labels = to_categorical(LE.fit_transform(labels))

    return data, encoded_labels

def create_audio_model():
    numClasses = 20
    model = Sequential()

    # Update the input shape to match the dimensions of your MFCC data
    model.add(InputLayer(input_shape=(1000,442,16)))

    model.add(Flatten())  # Flatten the input if it's not already a flat vector
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))

    return model

def main():
    # Load the audio data and labels
    mfcc_features, labels = audio_only_load_data()

    # Split the dataset into training, validation, and testing sets
    X_train, X_tmp, y_train, y_tmp = train_test_split(mfcc_features, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    # Create the audio model
    model = create_audio_model()

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Train the model
    num_epochs = 50
    num_batch_size = 64
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose=1)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100}%")

    # Optional: Plot training history, show confusion matrix, etc.

if __name__ == "__main__":
    main()
