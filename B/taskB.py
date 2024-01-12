import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam

# this script defines several function to implement Task B CNN model, including 
# load dataset, pre-processing dataset, define cnn model and train cnn model
def load_dataset(filepath):
    data = np.load(filepath)
    return (data[f] for f in ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels'])

def normalize_data(x_train, x_val, x_test):
    return (x.astype('float32') / 255.0 for x in [x_train, x_val, x_test])

def encode_labels(y_train, y_val, y_test, num_classes):
    return (tf.keras.utils.to_categorical(y, num_classes=num_classes) for y in [y_train, y_val, y_test])

def define_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (2, 2), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model, learning_rate):
    model.compile(loss="categorical_crossentropy", 
                  optimizer=Adam(lr=learning_rate), 
                  metrics=["accuracy"])
    return model

def train_model(model, x_train, y_train, epochs, batch_size, callbacks,x_val,y_val):
    return model.fit(x_train, y_train, 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     callbacks=callbacks,
                     validation_data=(x_val, y_val)
                     )

