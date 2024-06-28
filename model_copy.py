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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def audio_visual_ib_load_data():

    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        for subDir in child.glob('*'):
            binary_file = None
            mfcc_file = None
            for subsubDir in subDir.glob('*'):
                subsubDir = str(subsubDir)
                if 'ib' in subsubDir.split('_'):
                    binary_file = subsubDir
                elif 'mfcc' in subsubDir.split('_'):
                    mfcc_file = subsubDir

            if binary_file is not None and mfcc_file is not None:
                # print(binary_file)
                # print(mfcc_file)

                audio_feat = np.load(mfcc_file)
                visual_feat = np.load(binary_file)
                concat_audio_visual_ib_features = np.concatenate((audio_feat, visual_feat), axis=1)
                data.append(concat_audio_visual_ib_features)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    for i in range(0,20):
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


def create_model_ib_features():
    numClasses = 20
    model = Sequential()
    model.add(InputLayer(input_shape=(442, 20, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model


def time_stamp_formatter():
    current_time = datetime.datetime.today()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time


def main():
    audio_visual_ib_features, labels = audio_visual_ib_load_data()
    X_train_ib_features, X_tmp_ib_features, y_train_ib_features, y_tmp_ib_features = train_test_split(audio_visual_ib_features, labels, test_size=0.3, random_state=0)
    X_val_ib_features, X_test_ib_features, y_val_ib_features, y_test_ib_features = train_test_split(X_tmp_ib_features, y_tmp_ib_features, test_size=0.5, random_state=0)

    old_model_ib_features = create_model_ib_features()
    old_model_ib_features.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.005))

    # Uncomment below lines if you want to train the model
    num_epochs = 50
    num_batch_size = 32
    history_ib_features = old_model_ib_features.fit(X_train_ib_features, y_train_ib_features, validation_data=(X_val_ib_features, y_val_ib_features), batch_size=num_batch_size, epochs=num_epochs, verbose=1)
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

    predicted_probs_ib_features = old_model_ib_features.predict(X_test_ib_features, verbose=0)
    predicted_ib_features = np.argmax(predicted_probs_ib_features, axis=1)
    actual_ib_features = np.argmax(y_test_ib_features, axis=1)
    accuracy_ib_features = metrics.accuracy_score(actual_ib_features, predicted_ib_features)
    print(f'Accuracy: {accuracy_ib_features * 100}%')

    confusion_matrix_ib_features = metrics.confusion_matrix(np.argmax(y_test_ib_features, axis=1), predicted_ib_features)
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
