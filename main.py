from A.A_data_prep import load_data, preprocess_data, calculate_class_weights
from A.A_model import modelA
from A.stat import fbeta_score
from A.train import train_model
from keras.metrics import Precision, Recall

# Proceed with training and evaluation...


def taskA():

    # load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_data('Datasets/pneumoniamnist.npz')
    
    # preprocess dataset
    x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test)

    # calculate class weights
    class_weights = calculate_class_weights(y_train)
    # print(class_weights)
    # Example input shape and number of classes
    input_shape = (28, 28, 1) 
    num_classes = 2

    # create cnn model
    model = modelA(input_shape, num_classes)
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(name='precision'),
                           Recall(name='recall'), fbeta_score]
                )
    # train the model
    train_model(model,
                x_train, y_train,
                x_val, y_val,
                epochs=10,
                batch_size=32,
                class_weights=class_weights)
   


if __name__ == '__main__':
    taskA()
