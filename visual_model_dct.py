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

def visual_dct_load_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        for subDir in child.glob('*'):
            binary_file = None
            for subsubDir in subDir.glob('*'):
                subsubDir = str(subsubDir)
                if 'idct' in subsubDir.split('_'):
                    binary_file = subsubDir

            if binary_file is not None:
                visual_feat = np.load(binary_file)
                data.append(visual_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)

    data = np.array(data)
    for i in range(0,28):
        data[:,:,i] = data[:,:,i] / np.max(data[:,:,i])
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

def time_stamp_formatter():
    current_time = datetime.datetime.today()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time

def create_model_dct_features():
    numClasses = 20
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape = (442, 28, 1)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(InputLayer(input_shape=(442, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))
    return model


def main():
    # Load the visual data and labels
    visual_dct_features, labels = visual_dct_load_data()

    # Split the dataset into training, validation, and testing sets
    X_train_dct_features, X_tmp_dct_features, y_train_dct_features, y_tmp_dct_features = train_test_split(visual_dct_features, labels, test_size=0.3, random_state=0)
    X_val_dct_features, X_test_dct_features, y_val_dct_features, y_test_dct_features = train_test_split(X_tmp_dct_features, y_tmp_dct_features, test_size=0.5, random_state=0)

    # Create the model
    model = create_model_dct_features()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    # Uncomment below lines if you want to train the model

    # Train the model
    num_epochs = 100
    num_batch_size = 32
    history_dct_features = model.fit(X_train_dct_features, y_train_dct_features, validation_data=(X_val_dct_features, y_val_dct_features), batch_size=num_batch_size, epochs=num_epochs, verbose=1)
    # model.save_weights('adjustment_run3.h5')

    # # Load previously saved weights
    # # model.load_weights('final_run.h5')
    #
    # # Plotting and metrics
    #
    # Plot training history
    plt.plot(history_dct_features.history['accuracy'])
    plt.plot(history_dct_features.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.clf()

    # Evaluate the model on test data
    predicted_probs_dct_features = model.predict(X_test_dct_features, verbose=0)
    predicted_dct_features = np.argmax(predicted_probs_dct_features, axis=1)
    actual_dct_features = np.argmax(y_test_dct_features, axis=1)
    accuracy_dct_features = metrics.accuracy_score(actual_dct_features, predicted_dct_features)
    print(f'Accuracy: {accuracy_dct_features * 100}%')


    # Display the confusion matrix
    confusion_matrix_dct_features = metrics.confusion_matrix(actual_dct_features, predicted_dct_features)
    cm_display_dct_features = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dct_features)
    cm_display_dct_features.plot()
    plt.show()
    # plt.clf()
    # plt.savefig(time_stamp_formatter() + '_confusion_matrix.png')

    precision = precision_score(actual_dct_features, predicted_dct_features, average='weighted')
    recall = recall_score(actual_dct_features, predicted_dct_features, average='weighted')
    f1 = f1_score(actual_dct_features, predicted_dct_features, average='weighted')

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

    # Plot the loss history
    # Clip the loss values to remove extreme values
    train_loss_clipped = np.clip(history_dct_features.history['loss'], 0, 5)
    val_loss_clipped = np.clip(history_dct_features.history['val_loss'], 0, 5)

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