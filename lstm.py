from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

# Define the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_length))
model.add(LSTM(128, activation='relu'))
model.add(Dense(13, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()
