import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, GRU, Activation, Dropout, GlobalMaxPool1D, Conv1D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import re

df = pd.read_csv('data/train_val.csv')

def clean(text):
    # Remove "@" tags
    text = re.sub(r'@\w+', '', text)
    # Remove "http" tags
    text = re.sub(r'http\S+', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    text= emoji_pattern.sub(r'', text)
     # Remove emojis
     #text = emoji.get_emoji_regexp().sub('', text)
    
    return text

df['tweet'] = df['tweet'].apply(clean)
tweets = df['tweet'].values
labels = df['labels'].values


df['tweet'] = df['tweet'].apply(clean)
# Tokenize the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad sequences
max_length = 100  # Define the maximum length of your input tweets
padded_sequences = pad_sequences(sequences, maxlen=max_length)

all_labels = set()
for label in labels:
    label_list = label.split()
    all_labels.update(label_list)

num_labels = len(all_labels)
print(num_labels)

label_mapping = {label: i for i, label in enumerate(all_labels)}

encoded_labels = np.zeros((len(labels), num_labels), dtype=np.int32)
for i, label in enumerate(labels):
    label_indices = [label_mapping[l] for l in label.split(' ')]
    encoded_labels[i, label_indices] = 1
print(encoded_labels)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model.add(Dropout(0.6))
model.add(Conv1D(filters = 64, kernel_size = 10, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation='relu'))
#model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

# Evaluate the model
model.summary()
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

