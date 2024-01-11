from keras import layers, models


def modelA(input_shape, num_classes):
    """
    Create a CNN model.

    Parameters:
    input_shape (tuple): The shape of the input data (height, width, channels).
    num_classes (int): The number of classes for classification.

    Returns:
    keras.Model: A Keras CNN model.
    """

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    return model
