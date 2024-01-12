import numpy as np
import keras
from keras import layers
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from keras import backend as K
from keras.metrics import Precision, Recall

# This function includes all functions needed to implement Task A cnn model

def load_and_preprocess_data(filepath):
    data = np.load(filepath)
    x_train, y_train = data['train_images'], data['train_labels']
    x_val, y_val = data['val_images'], data['val_labels']
    x_test, y_test = data['test_images'], data['test_labels']

    # Normalize and reshape
    x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0
    x_train, x_val, x_test = [x.reshape((x.shape[0], 28, 28, 1)) for x in [x_train, x_val, x_test]]

    # Calculate class weights for imbalanced datasets
    class_weights = {
        0: (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0),
        1: (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0),
        }
    return x_train, y_train, x_val, y_val, x_test, y_test, class_weights

def fbeta_score(y_true, y_pred, beta=2):
    # Calculate the precision and recall using backend functions
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_true, 0, 1)))
    
    # Calculate the F-beta score
    bb = beta ** 2
    fbeta_score = (1 + bb) * (precision * recall) / (bb * precision + recall + K.epsilon())
    
    return fbeta_score

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile_and_train_model(model, x_train, y_train, x_val, y_val, class_weights):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), fbeta_score])
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val), class_weight=class_weights)
    return model, history

def evaluate_model(model, x_test, y_test):
    metrics = model.evaluate(x_test, y_test)
    test_loss, test_accuracy, test_precision, test_recall, test_fbeta = metrics
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test precision: {test_precision * 100:.2f}%")
    print(f"Test recall: {test_recall * 100:.2f}%")
    print(f"Test F2 score: {test_fbeta}")
    print(f"Test loss: {test_loss}")

    # Cohen's Kappa Score
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    kappa = cohen_kappa_score(y_test, y_pred_binary)
    print(f"Cohen's Kappa score: {kappa:.2f}")

def plot_results(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.ylim([0, 1])
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.ylim([0, 1])
    plt.show()













