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

import json
import datetime
from pathlib import Path


def visual_ib_load_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        for subDir in child.glob('*'):
            binary_file = None
            for subsubDir in subDir.glob('*'):
                subsubDir = str(subsubDir)
                if 'ib' in subsubDir.split('_'):
                    binary_file = subsubDir

            if binary_file is not None:
                # print(binary_file)

                visual_feat = np.load(binary_file)
                data.append(visual_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    for i in range(0,4):
        data[:,:,i] = data[:,:,i] / np.max(data[:,:,i])
    labels = np.array(labels)
    # print("Data shape:", data.shape)
    # print("Labels shape:", labels.shape)

    ##### our original label encoder #####
    # le = LabelEncoder().fit(labels)
    # encoded_labels = to_categorical(le.transform(labels))
    ######################################

    ##### label encoder used in lab example #####
    LE = LabelEncoder()
    encoded_labels = to_categorical(LE.fit_transform(labels))
    ############################################

    return data, encoded_labels

def create_model_visual_features():
    numClasses = 20
    model = Sequential()
    # model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape = (442, 4, 1)))

    model.add(InputLayer(input_shape=(442,4,1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(Activation('relu'))

    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model

def time_stamp_formatter():
    current_time = datetime.datetime.today()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time

def main():
    # Load visual data and labels
    visual_features, labels = visual_ib_load_data()

    # Splitting the dataset into training, validation, and test sets
    X_train, X_tmp, y_train, y_tmp = train_test_split(visual_features, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    # Create and compile the model
    model = create_model_visual_features()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))

    # Uncomment below lines if you want to train the model
    num_epochs = 100
    num_batch_size = 32
    history_ib_features = model.fit(X_train, y_train, validation_data=(X_val, y_val),
     batch_size=num_batch_size, epochs=num_epochs, verbose=1)
    # model.save_weights('adjustment_run3.h5')

    # # Load previously saved weights
    # # model.load_weights('final_run.h5')
    #
    # # Plotting and metrics
    #
    # Uncomment this section if you have trained the model
    plt.plot(history_ib_features.history['accuracy'])
    plt.plot(history_ib_features.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.clf()

    # Evalute model on test data
    predicted_probs_ib_features = model.predict(X_test, verbose=0)
    predicted_ib_features = np.argmax(predicted_probs_ib_features, axis=1)
    actual_ib_features = np.argmax(y_test, axis=1)
    accuracy_ib_features = metrics.accuracy_score(actual_ib_features, predicted_ib_features)
    print(f'Accuracy: {accuracy_ib_features * 100}%')

    confusion_matrix_ib_features = metrics.confusion_matrix(np.argmax(y_test, axis=1),
                                                            predicted_ib_features)
    cm_display_ib_features = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_ib_features)
    cm_display_ib_features.plot()
    plt.show()
    plt.clf()
    # plt.savefig(time_stamp_formatter() + '_confusion_matrix.png')

    precision = precision_score(actual_ib_features, predicted_ib_features, average='weighted')
    recall = recall_score(actual_ib_features, predicted_ib_features, average='weighted')
    f1 = f1_score(actual_ib_features, predicted_ib_features, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Plot Precision, Recall, and F1 Score
    metrics_values = [precision, recall, f1]
    metric_names = ['Precision', 'Recall', 'F1 Score']

    plt.bar(metric_names, metrics_values)
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Precision, Recall and F1 Score')
    plt.ylim(0, 1)
    plt.show()
    plt.clf()

    # Clip the loss values to remove extreme values
    train_loss_clipped = np.clip(history_ib_features.history['loss'], 0, 5)
    val_loss_clipped = np.clip(history_ib_features.history['val_loss'], 0, 5)

    # Plot the clipped loss values
    plt.plot(train_loss_clipped, label='Train')
    plt.plot(val_loss_clipped, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()