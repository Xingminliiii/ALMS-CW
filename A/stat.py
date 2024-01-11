from keras import backend as K
from keras.metrics import Precision, Recall


def fbeta_score(y_true, y_pred, beta=2):
    # Calculate the precision and recall using backend functions
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_true, 0, 1)))
    
    # Calculate the F-beta score
    bb = beta ** 2
    fbeta_score = (1 + bb) * (precision * recall) / (bb * precision + recall + K.epsilon())
    
    return fbeta_score