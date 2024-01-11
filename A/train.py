from keras.callbacks import EarlyStopping

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, class_weights=None):
    """
    Train the CNN model.

    Parameters:
    model (keras.Model): The CNN model to train.
    x_train (numpy.ndarray): Training data.
    y_train (numpy.ndarray): Training labels.
    x_val (numpy.ndarray): Validation data.
    y_val (numpy.ndarray): Validation labels.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.
    class_weights (dict, optional): Weights for classes in case of imbalanced dataset.

    Returns:
    history: Training history object containing training and validation loss and accuracy.
    """

    # Training the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
    )

    return history
