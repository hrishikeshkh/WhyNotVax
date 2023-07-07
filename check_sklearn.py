#sc check sklearn's version mismatch 

import sklearn
print(sklearn.__version__)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
#import MLP module from tf
from tensorflow.python.keras import models


import numpy as np


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


#create a model with a basic NN (assume values for input/output)
model = mlp_model(
        layers=10,
        units=5,
        dropout_rate=0.2,
        input_shape=x_train.shape[1:],
        num_classes=2,
    )

model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, verbose=1)
print(history.params)

# check the keys of history object
print(history.history.keys())
