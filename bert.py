import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Load data from CSV file
data = pd.read_csv('data/train_val_cleaned.csv')

# Preprocess the data
tweets = data['tweet'].tolist()
labels = data['labels'].tolist()

# Tokenize the tweets using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.encode(tweet, add_special_tokens=True, max_length=128, truncation=True) for tweet in tweets]

# Pad the tokenized sequences to the same length
max_length = max(len(seq) for seq in tokenized_texts)
input_ids = tf.keras.preprocessing.sequence.pad_sequences(tokenized_texts, maxlen=max_length, padding='post')

# Convert labels to one-hot encoded vectors
all_labels = set()
for label in labels:
    label_list = label.split()
    all_labels.update(label_list)

num_labels = len(all_labels)
label_mapping = {label: i for i, label in enumerate(all_labels)}
labels_encoded = np.zeros((len(labels), num_labels), dtype=np.int32)
for i, label in enumerate(labels):
    label_indices = [label_mapping[l] for l in label.split()]
    labels_encoded[i, label_indices] = 1

# Define the BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define the architecture of the multi-label classification model
inputs = tf.keras.layers.Input(shape=(max_length,), dtype='int32')
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype='int32')
bert_output = bert_model(inputs, attention_mask=attention_mask)[0]
# CHANGED: Use 'sigmoid' activation for multi-label classification
output = tf.keras.layers.Dense(num_labels, activation='sigmoid')(bert_output)

model = tf.keras.models.Model(inputs=[inputs, attention_mask], outputs=output)

# Define the loss function and compile the model
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=loss)

# Train the model
model.fit([input_ids, attention_mask], labels_encoded, epochs=15, batch_size=64)

# Inference
def predict_labels(tweet):
    # Tokenize and pad the tweet
    tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=True, truncation=True)
    padded_tweet = tf.keras.preprocessing.sequence.pad_sequences([tokenized_tweet], maxlen=max_length, padding='post')
    
    # Create an attention mask
    attention_masks = np.where(padded_tweet != 0, 1, 0)
    
    predictions = model.predict([padded_tweet, attention_masks])
    return predictions

# Example usage
tweet = "I don't want to take the vaccine because it is very dangerous to my lungs"
predictions = predict_labels(tweet)
