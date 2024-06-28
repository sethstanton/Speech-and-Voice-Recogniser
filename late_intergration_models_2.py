from timeit import default_timer as Timer
import numpy as np
from matplotlib import pyplot as plt
import json
import datetime
import torch.nn.functional as F
from pathlib import Path
import torch 
import torchmetrics
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torch import nn ## torch nueral networks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_confusion_matrix
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LE = LabelEncoder()

def load_binary_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        # print(child)
        for subDir in child.glob('*'):
            # print(subDir)
            binary_file = None
            for subsubDir in subDir.glob('*'):
                    # print(subsubDir)
                    subsubDir = str(subsubDir)

                    if 'binaryarray' in subsubDir.split('_'):
                        binary_file = subsubDir
            
            if binary_file != None:
                # print(binary_file)
                visual_feat = np.load(binary_file)
                data.append(visual_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    encoded_labels = LE.fit_transform(labels)

    return torch.from_numpy(data).type(torch.float), torch.from_numpy(encoded_labels).type(torch.long)

def load_dct_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        # print(child)
        for subDir in child.glob('*'):
            # print(subDir)
            dct_file = None

            for subsubDir in subDir.glob('*'):
                    # print(subsubDir)
                    subsubDir = str(subsubDir)
                    ### change this to 'idct' if you want to use idct audio_visual_ib_features ###
                    if 'dctarray' in subsubDir.split('_'):
                        dct_file = subsubDir
            
            if dct_file != None:
                # print(dct_file)
                visual_feat = np.load(dct_file)
                data.append(visual_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    encoded_labels = LE.fit_transform(labels)
 
    return torch.from_numpy(data).type(torch.float), torch.from_numpy(encoded_labels).type(torch.long)

def load_mfcc_data():
    data = []
    labels = []

    DIR_PATH = Path('recordings/')
    for child in DIR_PATH.glob('*'):
        # print(child)
        for subDir in child.glob('*'):
            # print(subDir)
            mfcc_file = None

            for subsubDir in subDir.glob('*'):
                    # print(subsubDir)
                    subsubDir = str(subsubDir)
                    ### change this to 'idct' if you want to use idct audio_visual_ib_features ###
                    if 'mfcc' in subsubDir.split('_'):
                        mfcc_file = subsubDir
            
            if mfcc_file != None:
                # print(mfcc_file)
                audio_feat = np.load(mfcc_file)
                data.append(audio_feat)
                label = str(subDir).split('\\')[1]
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    encoded_labels = LE.fit_transform(labels)
 
    return torch.from_numpy(data).type(torch.float), torch.from_numpy(encoded_labels).type(torch.long)


def tts_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

class AudioModel(nn.Module):
    def __init__(self, hidden_units, output_size):
        super(AudioModel, self).__init__()

        # Assuming input_shape is (800, 442, 16)
        self.conv1 = nn.Conv2d(16, hidden_units, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_units, hidden_units, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_units * 400 * 8, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# class AudioModel(nn.Module):
#     def __init__(self, input_shape, hidden_units, output_size):
#         super(AudioModel, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(hidden_units * (input_shape[1] // 2) * (input_shape[2] // 2), output_size)
#         self.relu3 = nn.ReLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         return x

class VisualModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_size):
        super(VisualModel, self).__init__()

        self.fc1 = nn.Linear(input_shape[1] * input_shape[2], hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class LateIntegrationModel(nn.Module):
    def __init__(self, visual_input_shape, hidden_units, output_size):
        super(LateIntegrationModel, self).__init__()

        # Audio Model
        self.audio_model = AudioModel( hidden_units=hidden_units, output_size=output_size)

        # Visual Model
        # visual_input_size = visual_input_shape[1] * visual_input_shape[2]
        self.visual_model = VisualModel(input_shape=visual_input_shape, hidden_units=hidden_units, output_size=output_size)

        # Fusion Layer
        self.fusion_layer = nn.Linear(2 * output_size, output_size)  # Assuming concatenation of audio and visual features

    def forward(self, audio_input, visual_input):
        # Forward pass for audio model
        audio_output = self.audio_model(audio_input)

        # Forward pass for visual model
        visual_output = self.visual_model(visual_input.view(visual_input.size(0), -1))

        # Concatenate audio and visual features
        concatenated_features = torch.cat((audio_output, visual_output), dim=1)

        # Apply fusion layer
        fused_output = self.fusion_layer(concatenated_features)

        return F.log_softmax(fused_output, dim=1)  # Assuming a classification task, applying log_softmax
        
# class LateIntegrationModel(nn.Module):
#     def __init__(self, audio_input_shape, visual_input_shape, hidden_units, output_size):
#         super(LateIntegrationModel, self).__init__()

#         # Audio Model
#         self.audio_model = AudioModel(input_shape=audio_input_shape, hidden_units=hidden_units, output_size=output_size)

#         # Visual Model
#         self.visual_model = VisualModel(input_shape=visual_input_shape, hidden_units=hidden_units, output_size=output_size)

#         # Fusion Layer
#         self.fusion_layer = nn.Linear(2 * output_size, output_size)  # Assuming concatenation of audio and visual features

#     def forward(self, audio_input, visual_input):
#         # Forward pass for audio model
#         audio_output = self.audio_model(audio_input)

#         # Forward pass for visual model
#         visual_output = self.visual_model(visual_input)

#         # Concatenate audio and visual features
#         concatenated_features = torch.cat((audio_output, visual_output), dim=1)

#         # Apply fusion layer
#         fused_output = self.fusion_layer(concatenated_features)

#         return fused_output

def train_step(model, X_train, y_train, loss_fn, optimizer, accuracy_fn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Performs a training step with the model trying to learn on X_train, y_train
    """
    model = model.to(device)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.long).to(device)

    # Put the model into training mode
    model.train()

    y_pred = model(X_train)

    # Calculate the loss and accuracy
    loss = loss_fn(y_pred, y_train)
    train_acc = accuracy_fn(y_true=y_train, y_pred=y_pred.argmax(dim=1))

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f'Train loss: {loss.item():.5f} | Train acc: {train_acc.item():.2f}%\n')

    return {'train_loss': loss.item(),
            'train_acc': train_acc.item()}

def test_step(model, X_test, y_test, loss_fn, accuracy_fn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Computes test loss and accuracy on the given test data (X_test, y_test)
    """
    model = model.to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.long).to(device)

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)

        test_loss = loss_fn(test_pred, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

    print(f'Test loss: {test_loss.item():.5f} | Test acc: {test_acc.item():.2f}%\n')
    return {'test_loss': test_loss.item(),
            'test_acc': test_acc.item()}

def train_step2(model, audio_data, visual_data, labels, loss_fn, optimizer, accuracy_fn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Performs a training step with the LateFusionModel trying to learn on audio_data, visual_data, labels
    """
    model = model.to(device)
    audio_data, visual_data, labels = (
        torch.tensor(audio_data, dtype=torch.float32).to(device),
        torch.tensor(visual_data, dtype=torch.float32).to(device),
        torch.tensor(labels, dtype=torch.long).to(device)
    )

    # Put the model into training mode
    model.train()

    # Forward pass
    output = model(audio_data, visual_data)

    # Calculate the loss and accuracy
    loss = loss_fn(output, labels)
    train_acc = accuracy_fn(y_true=labels, y_pred=output.argmax(dim=1))

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f'Train loss: {loss.item():.5f} | Train acc: {train_acc.item():.2f}%\n')
    return {'train_loss': loss.item(),
            'train_acc': train_acc.item()}

def test_step2(model, audio_data, visual_data, labels, loss_fn, accuracy_fn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Computes test loss and accuracy on the given audio_data, visual_data, labels
    """
    model = model.to(device)
    audio_data, visual_data, labels = (
        torch.tensor(audio_data, dtype=torch.float32).to(device),
        torch.tensor(visual_data, dtype=torch.float32).to(device),
        torch.tensor(labels, dtype=torch.long).to(device)
    )

    # Put the model into evaluation mode
    model.eval()

    # Forward pass
    output = model(audio_data, visual_data)

    # Calculate the loss and accuracy
    loss = loss_fn(output, labels)
    test_acc = accuracy_fn(y_true=labels, y_pred=output.argmax(dim=1))

    print(f'Test loss: {loss.item():.5f} | Test acc: {test_acc.item():.2f}%\n')
    return {'test_loss': loss.item(),
            'test_acc': test_acc.item()}

def eval_model(model, 
               X, y, 
               loss_fn, 
               accuracy_fn, 
               device=device):
    """
    Returns a dictionary containing the results of model predicting on X, y
    """
    model = model.to(device)
    X, y = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item(),
            'model_acc': acc.item()}

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """
    Prints difference between start and end time
    
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    
    """
    total_time = end-start
    print(f'Training time on {device}: {total_time:.3f} seconds')
    return total_time

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def plot_conf_matrix(model, data, labels, title, device=device):
    """
    Plots a confusion matrix for the given data and labels using the provided model.
    """
    model = model.to(device)
    data, labels = (
        torch.tensor(data, dtype=torch.float32).to(device),
        torch.tensor(labels, dtype=torch.long).to(device)
    )

    # Put the model into evaluation mode
    model.eval()

    # Forward pass
    output = model(data)

    # Calculate predictions and convert to class labels
    predictions = torch.argmax(output, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    # Calculate confusion matrix using torchmetrics
    conf_matrix_metric = ConfusionMatrix(num_classes=20, task='multiclass')
    conf_matrix_metric.update(predictions, true_labels)
    conf_matrix = conf_matrix_metric.compute().cpu().numpy()

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                    show_absolute=True,
                                    show_norm=True,
                                    colorbar=True)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title + ' Confusion Matrix')
    plt.show()

def plot_conf_matrix2(model, audio_data, visual_data, labels, title, device=device):
    """
    Plots a confusion matrix for the given audio_data, visual_data, and labels using the provided model.
    """
    model = model.to(device)
    audio_data, visual_data, labels = (
        torch.tensor(audio_data, dtype=torch.float32).to(device),
        torch.tensor(visual_data, dtype=torch.float32).to(device),
        torch.tensor(labels, dtype=torch.long).to(device)
    )

    # Put the model into evaluation mode
    model.eval()

    # Forward pass
    output = model(audio_data, visual_data)

    # Calculate predictions and convert to class labels
    predictions = torch.argmax(output, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    # Calculate confusion matrix using torchmetrics
    conf_matrix_metric = ConfusionMatrix(num_classes=20, task='multiclass')
    conf_matrix_metric.update(predictions, true_labels)
    conf_matrix = conf_matrix_metric.compute().cpu().numpy()

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                    show_absolute=True,
                                    show_norm=True,
                                    colorbar=True)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title + ' Confusion Matrix')
    plt.show()

binary_x, binary_y = load_binary_data()
dct_x, dct_y = load_dct_data()
audio_x, audio_y = load_mfcc_data()

binary_X_train, binary_X_test, binary_y_train, binary_y_test = tts_data(binary_x, binary_y)

dct_X_train, dct_X_test, dct_y_train, dct_y_test = tts_data(dct_x, dct_y)

audio_X_train, audio_X_test, audio_y_train, audio_y_test = tts_data(audio_x, audio_y)

audio_model = AudioModel( hidden_units=32, output_size=20)
binary_visual_model = VisualModel(input_shape=(1,110, 4), hidden_units=32, output_size=20)
dct_visual_model = VisualModel(input_shape=(1,110, 26), hidden_units=32, output_size=20)
audio_binary_visual_model = LateIntegrationModel( visual_input_shape=(1, 110, 4), hidden_units=32, output_size=20)
audio_dct_visual_model = LateIntegrationModel( visual_input_shape=(1, 110, 26), hidden_units=32, output_size=20)
loss_fn = nn.CrossEntropyLoss()
acc_fn = accuracy_fn

audio_optimizer = torch.optim.SGD(params=audio_model.parameters(),
                                  lr=0.1)
binary_visual_optimizer = torch.optim.SGD(params=binary_visual_model.parameters(),
                                          lr=0.1)
dct_visual_optimizer = torch.optim.SGD(params=dct_visual_model.parameters(),
                                       lr=0.1)
audio_binary_visual_optimizer = torch.optim.SGD(params=audio_binary_visual_model.parameters(),
                                                lr=0.1)
audio_dct_visual_optimizer = torch.optim.SGD(params=audio_dct_visual_model.parameters(),
                                             lr=0.1)

audio_results = {'train_loss': [],
                 'train_acc': [],
                 'test_loss': [],
                 'test_acc': []}
binary_visual_results = {'train_loss': [],
                         'train_acc': [],
                         'test_loss': [],
                         'test_acc': []}
dct_visual_results = {'train_loss': [],
                        'train_acc': [],
                        'test_loss': [],
                        'test_acc': []}
audio_binary_visual_results = {'train_loss': [],
                                 'train_acc': [],
                                 'test_loss': [],
                                 'test_acc': []}
audio_dct_visual_results = {'train_loss': [],
                                'train_acc': [],
                                'test_loss': [],
                                'test_acc': []}

train_time_start_on_gpu = Timer()
epochs = 20
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    audio_results.update(train_step(model=audio_model,
                                    X_train=audio_X_train,
                                    y_train=audio_y_train,
                                    loss_fn=loss_fn,
                                    optimizer=audio_optimizer,
                                    accuracy_fn=acc_fn))
    
    audio_results.update(test_step(model=audio_model,
                                   X_test=audio_X_test,
                                   y_test=audio_y_test,
                                   loss_fn=loss_fn,
                                   accuracy_fn=acc_fn))
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')    
    binary_visual_results.update(train_step(model=binary_visual_model,
                                            X_train=binary_X_train,
                                            y_train=binary_y_train,
                                            loss_fn=loss_fn,
                                            optimizer=binary_visual_optimizer,
                                            accuracy_fn=acc_fn))
    
    binary_visual_results.update(test_step(model=binary_visual_model,
                                           X_test=binary_X_test,
                                           y_test=binary_y_test,
                                           loss_fn=loss_fn,
                                           accuracy_fn=acc_fn))
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')    
    dct_visual_results.update(train_step(model=dct_visual_model,
                                         X_train=dct_X_train,
                                         y_train=dct_y_train,
                                         loss_fn=loss_fn,
                                         optimizer=dct_visual_optimizer,
                                         accuracy_fn=acc_fn))
    
    dct_visual_results.update(test_step(model=dct_visual_model,
                                        X_test=dct_X_test,
                                        y_test=dct_y_test,
                                        loss_fn=loss_fn,
                                        accuracy_fn=acc_fn))
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')    
    audio_binary_visual_results.update(train_step2(model=audio_binary_visual_model,
                                                    audio_data=audio_X_train,
                                                    visual_data=binary_X_train,
                                                    labels=binary_y_train,
                                                    loss_fn=loss_fn,
                                                    optimizer=audio_binary_visual_optimizer,
                                                    accuracy_fn=acc_fn))
    
    audio_binary_visual_results.update(test_step2(model=audio_binary_visual_model,
                                                    audio_data=audio_X_test,
                                                    visual_data=binary_X_test,
                                                    labels=binary_y_test,
                                                    loss_fn=loss_fn,
                                                    accuracy_fn=acc_fn))
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')    
    audio_dct_visual_results.update(train_step2(model=audio_dct_visual_model,
                                                audio_data=audio_X_train,
                                                visual_data=dct_X_train,
                                                labels=dct_y_train,
                                                loss_fn=loss_fn,
                                                optimizer=audio_dct_visual_optimizer,
                                                accuracy_fn=acc_fn))
    
    audio_dct_visual_results.update(test_step2(model=audio_dct_visual_model,
                                                audio_data=audio_X_test,
                                                visual_data=dct_X_test,
                                                labels=dct_y_test,
                                                loss_fn=loss_fn,
                                                accuracy_fn=acc_fn))
    
train_time_end_on_gpu = Timer()
total_train_time_model_1_gpu = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

audio_model_eval_results = eval_model(model=audio_model,
                                        X=audio_X_test,
                                        y=audio_y_test,
                                        loss_fn=loss_fn,
                                        accuracy_fn=acc_fn)
binary_visual_model_eval_results = eval_model(model=binary_visual_model,
                                                X=binary_X_test,
                                                y=binary_y_test,
                                                loss_fn=loss_fn,
                                                accuracy_fn=acc_fn)
dct_visual_model_eval_results = eval_model(model=dct_visual_model,
                                            X=dct_X_test,
                                            y=dct_y_test,
                                            loss_fn=loss_fn,
                                            accuracy_fn=acc_fn)
audio_binary_visual_model_eval_results = eval_model(model=audio_binary_visual_model,
                                                    X=audio_X_test,
                                                    y=binary_y_test,
                                                    loss_fn=loss_fn,
                                                    accuracy_fn=acc_fn)
audio_dct_visual_model_eval_results = eval_model(model=audio_dct_visual_model,
                                                    X=audio_X_test,
                                                    y=dct_y_test,
                                                    loss_fn=loss_fn,
                                                    accuracy_fn=acc_fn)

print(f'Audio Model: {audio_model_eval_results}\n')
print(f'Binary Visual Model: {binary_visual_model_eval_results}\n')
print(f'DCT Visual Model: {dct_visual_model_eval_results}\n')
print(f'Audio Binary Visual Model: {audio_binary_visual_model_eval_results}\n')
print(f'Audio DCT Visual Model: {audio_dct_visual_model_eval_results}\n')

plot_loss_curves(audio_results)
plot_loss_curves(binary_visual_results)
plot_loss_curves(dct_visual_results)
plot_loss_curves(audio_binary_visual_results)
plot_loss_curves(audio_dct_visual_results)

plot_conf_matrix(model=audio_model,
                    data=audio_X_test,
                    labels=audio_y_test,
                    title='Audio Model')
plot_conf_matrix(model=binary_visual_model,
                    data=binary_X_test,
                    labels=binary_y_test,
                    title='Binary Visual Model')
plot_conf_matrix(model=dct_visual_model,
                    data=dct_X_test,
                    labels=dct_y_test,
                    title='DCT Visual Model')
plot_conf_matrix2(model=audio_binary_visual_model,
                    audio_data=audio_X_test,
                    visual_data=binary_X_test,
                    labels=binary_y_test,
                    title='Audio Binary Visual Model')
plot_conf_matrix2(model=audio_dct_visual_model,
                    audio_data=audio_X_test,
                    visual_data=dct_X_test,
                    labels=dct_y_test,
                    title='Audio DCT Visual Model')



























