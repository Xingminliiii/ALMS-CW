import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc


def B_evaluate_model(model, x_test, y_test, x_val, y_val):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {val_accuracy}")

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim([0, 1])
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim([0, 1])
    plt.show()

def generate_classification_report(model, x_test, y_test, batch_size, class_names):
    test_samples = x_test.shape[0]
    steps = np.ceil(test_samples / batch_size)

    res_predictions = model.predict(x_test, steps=steps, verbose=1)
    res_pred_labels = np.argmax(res_predictions, axis=1)

    print("Shape of predictions:", res_predictions.shape)
    print("Length of predicted labels:", len(res_pred_labels))
    print("Length of actual labels:", len(y_test))

    print('|' + '-' * 67 + '|')
    print('|-------Classification Report: MNIST_CNN Training Cycle #1----------|')
    print('|' + '-' * 67 + '|')
    print(classification_report(y_test.argmax(axis=1), res_pred_labels, target_names=class_names))

def plot_confusion_matrix(y_test, predictions, target_names):
    cm = confusion_matrix(y_test.argmax(axis=1), predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

def plot_roc_curves(y_test, predictions, class_names):
    num_classes = len(class_names)
    true_labels = np.argmax(y_test, axis=1) if y_test.shape[1] > 1 else y_test

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((true_labels == i).astype(int), predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-Class')
    plt.legend(loc="lower right")
    plt.show()


