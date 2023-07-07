#sc check sklearn's version mismatch 

import sklearn
print(sklearn.__version__)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import numpy as np
#create a model with a basic NN (assume values for input/output)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, verbose=1)
print(history.params)

# check the keys of history object
print(history.history.keys())
