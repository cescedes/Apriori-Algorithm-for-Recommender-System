# Solution for task 2 (Image Classifier) of lab assignment for FDA SS23 by Ş. Aybüke

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

def train_predict(X_train, y_train, X_test):

    assert X_train.shape == (len(X_train), 6336)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 6336)

    n_classes = 40  # checked np.unique(y) in jupyter notebook to decide n_classes
    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train_cat = to_categorical(y_train)

    np.random.seed(245)
    tf.random.set_seed(3754)

    model = models.Sequential()

    model.add(layers.Dense(160, activation='relu', input_shape=X_train.shape[1:]))

    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.01),
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, epochs=100)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)

    return y_pred


if __name__ == "__main__":
    # load data (please load data like that and let every processing step happen **inside** the train_predict function)
    # (change path if necessary)
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    # please put everything that you want to execute outside the function here!


