import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Dropout, Conv1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load and preprocess data
df = pd.read_csv('data/train_val_cleaned.csv')
tweets = df['tweet'].values
labels = df['labels'].values

# Use the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 100
tokenized_texts = [tokenizer.encode(tweet, add_special_tokens=True, max_length=max_length, truncation=True) for tweet in tweets]
padded_sequences = np.array(pad_sequences(tokenized_texts, maxlen=max_length, dtype="long", value=0))

all_labels = set()
for label in labels:
    label_list = label.split()
    all_labels.update(label_list)

num_labels = len(all_labels)
label_mapping = {label: i for i, label in enumerate(all_labels)}
encoded_labels = np.zeros((len(labels), num_labels), dtype=np.int32)
for i, label in enumerate(labels):
    label_indices = [label_mapping[l] for l in label.split(' ')]
    encoded_labels[i, label_indices] = 1

# Create attention masks
attention_masks = (padded_sequences > 0).astype(int)

# Split the data into train and test sets
X_train_ids, X_test_ids, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
X_train_masks, X_test_masks, _ , _ = train_test_split(attention_masks, encoded_labels, test_size=0.2, random_state=42)

# Define the integrated BERT-LSTM model
input_ids_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
attention_masks_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_output = bert_model([input_ids_layer, attention_masks_layer])[0]

lstm_out = Dropout(0.6)(bert_output)
lstm_out = Conv1D(filters=50, kernel_size=20, padding='valid', activation='relu', strides=2)(lstm_out)
lstm_out = LSTM(128, activation='relu')(lstm_out)
final_output = Dense(num_labels, activation='sigmoid')(lstm_out)

model = tf.keras.Model(inputs=[input_ids_layer, attention_masks_layer], outputs=final_output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([X_train_ids, X_train_masks], y_train, validation_data=([X_test_ids, X_test_masks], y_test), epochs=10, batch_size=16)

# Evaluate the model
model.summary()
loss, accuracy = model.evaluate([X_test_ids, X_test_masks], y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
