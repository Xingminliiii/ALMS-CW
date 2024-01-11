import numpy as np


def load_data(file_path):
    data = np.load(file_path)
    return data['train_images'], data['train_labels'], data['val_images'], \
        data['val_labels'], data['test_images'], data['test_labels']


def preprocess_data(x_train, x_val, x_test):
    # Normalize pixel values to be between 0 and 1
    x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

    # Reshape images to indicate single color channel
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_val = x_val.reshape((x_val.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    return x_train, x_val, x_test


def calculate_class_weights(y_train):
    # Class weights calculation 
    class_weights = {
        0: (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0),
        1: (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0),
    }
    return class_weights
