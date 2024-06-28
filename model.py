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


def load_data():
    mfcc_features = np.load('mfcc_features.npy')
    with open('audio_names.json', 'r') as file:
        data = json.load(file)
    names_list = list(data.keys())
    labels = [name for name in names_list for _ in range(20)]
    labels = np.array(labels)
    le = LabelEncoder().fit(labels)
    encoded_labels = to_categorical(le.transform(labels))
    return mfcc_features, encoded_labels


def create_model():
    numClasses = 20
    model = Sequential()
    model.add(InputLayer(input_shape=(298, 16, 1)))
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
    features, labels = load_data()
    X_train, X_tmp, y_train, y_tmp = train_test_split(features, labels, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    model = create_model()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.005))

    # Uncomment below lines if you want to train the model
    num_epochs = 20
    num_batch_size = 32
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose=1)
    # model.save_weights('adjustment_run3.h5')

    # # Load previously saved weights
    # # model.load_weights('final_run.h5')
    #
    # # Plotting and metrics
    #
    # Uncomment this section if you have trained the model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.clf()

    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print(f'Accuracy: {accuracy * 100}%')

    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.show()
    plt.clf()
    plt.savefig(time_stamp_formatter() + '_confusion_matrix.png')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
