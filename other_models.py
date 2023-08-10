from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, GRU, Activation, Dropout, GlobalMaxPool1D, Conv1D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

# ... [Your previous preprocessing code remains unchanged]
df = pd.read_csv('data/train_val_cleaned.csv')

tweets = df['tweet'].values
labels = df['labels'].values

# Tokenize the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad sequences
max_length = 60  # Define the maximum length of your input tweets
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

# 

# 1. Decision Tree Classifier
# clf_tree = DecisionTreeClassifier(random_state=42)
# clf_tree.fit(X_train, y_train)
# y_pred_tree = clf_tree.predict(X_test)
# print("Decision Tree Classifier Results:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
# print(classification_report(y_test, y_pred_tree))

# 2. Support Vector Machine with class weights
clf_svm = OneVsRestClassifier(SVC(class_weight='balanced', kernel='linear', probability=True, random_state=42))
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print("\nSupport Vector Machine Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

# 3. Random Forest Classifier with class weights
# clf_rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
# clf_rf.fit(X_train, y_train)
# y_pred_rf = clf_rf.predict(X_test)
# print("\nRandom Forest Classifier Results:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
# print(classification_report(y_test, y_pred_rf))
