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


#create a MLP model
#with random values of choice

model = models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46, activation='softmax'))

#decalre an optimizer to adam
optimizer = tf.keras.optimizers.Adam(lr=0.001)

#compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

