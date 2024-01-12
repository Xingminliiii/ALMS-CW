from A.taskA import load_and_preprocess_data, build_model, \
    compile_and_train_model, evaluate_model, plot_results

from B.taskB import load_dataset, normalize_data, encode_labels, \
    define_model, compile_model, train_model
from B.evaluation import B_evaluate_model, plot_training_history, \
    generate_classification_report, plot_confusion_matrix, plot_roc_curves
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def TASKA():
    # function to perform Task A CNN model and evaluate its performance

    # Load file 
    filepath = 'Datasets/pneumoniamnist.npz'

    # Preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test,\
        class_weights = load_and_preprocess_data(filepath)
    
    # Build Model
    model = build_model()
    
    # Compile and train model
    model, history = compile_and_train_model(model, x_train, y_train, x_val, y_val, class_weights)
    
    #Evaluate model 
    evaluate_model(model, x_test, y_test)
    plot_results(history)

def TASKB():
    # function to perform Task A CNN model and evaluate its performance

    # Load file and define hyperparameteter
    filepath = 'Datasets/pathmnist.npz'
    model_path = 'B/MNIST_CNN.h5'
    num_classes = 9
    input_shape = (28, 28, 3)
    learning_rate = 0.001
    epochs = 50
    batch_size = 128

    # Preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(filepath)
    x_train, x_val, x_test = normalize_data(x_train, x_val, x_test)
    y_train, y_val, y_test = encode_labels(y_train, y_val, y_test, num_classes)

    model = define_model(input_shape, num_classes)
    model = compile_model(model, learning_rate)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4),
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    ]
    # Train the model
    history = train_model(model, x_train, y_train, epochs, batch_size, callbacks,x_val,y_val)
    
    # Evaluate the model
    B_evaluate_model(model, x_test, y_test, x_val, y_val)

    # Plot the training history
    plot_training_history(history)
    
    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    generate_classification_report(model, x_test, y_test, batch_size=1, class_names=class_names)

    # Predictions for confusion matrix and ROC curves
    predictions = model.predict(x_test)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, np.argmax(predictions, axis=1), class_names)

    # Plot ROC Curves
    plot_roc_curves(y_test, predictions, class_names)


if __name__ == '__main__':
    TASKA()
    TASKB()
