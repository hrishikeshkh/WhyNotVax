# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Reading Data
df = pd.read_csv("data/super_clean.csv")
X_text = df["tweet"]
y_multilabel = df[
    [
        "conspiracy",
        "country",
        "ineffective",
        "ingredients",
        "mandatory",
        "none",
        "pharma",
        "political",
        "religious",
        "rushed",
        "side-effect",
        "unnecessary",
    ]
]

# Step 2: Data Splitting
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_multilabel, random_state=42
)

# Step 3: Text Vectorization using Tokenizer and TF-IDF weighting
vocab_size = 20000  # Assuming a vocabulary size of 20k
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Padding sequences
max_len = max(max(len(seq) for seq in X_train_seq), max(len(seq) for seq in X_test_seq))
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# TF-IDF weighting
tfidf_weights = {word: idf for word, idf in zip(tokenizer.word_index.keys(), tokenizer.word_index.values())}
X_train_tfidf = [[tfidf_weights[word] if word in tfidf_weights else 0 for word in seq] for seq in X_train_seq]
X_test_tfidf = [[tfidf_weights[word] if word in tfidf_weights else 0 for word in seq] for seq in X_test_seq]

# Converting to TensorFlow tensors
X_train_tfidf = tf.convert_to_tensor(X_train_tfidf)
X_test_tfidf = tf.convert_to_tensor(X_test_tfidf)

# Step 5: Model Definition with Feature Selection
class ClassifierChainModel(tf.keras.Model):
    def __init__(self, num_labels, num_features):
        super(ClassifierChainModel, self).__init__()
        self.feature_selector = tf.keras.layers.Dense(num_features, activation="relu")  # Feature selection layer
        self.mlp_layers = [
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ]
        self.classifiers = [tf.keras.layers.Dense(1, activation="sigmoid") for _ in range(num_labels)]

    def call(self, inputs):
        x = self.feature_selector(inputs)  # Apply feature selection
        for layer in self.mlp_layers:
            x = layer(x)
        outputs = [classifier(x) for classifier in self.classifiers]
        return tf.concat(outputs, axis=-1)

# Create a Classifier Chain with feature selection
num_features = 2000  # You can adjust this to select a different number of features
num_labels = y_train.shape[1]
model = ClassifierChainModel(num_labels, num_features)

# Step 6: Training
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

# Step 7: Prediction and Evaluation
y_pred = model.predict(X_test_tfidf)
# Evaluate the model using additional metrics if needed

# Step 8: Model Saving
model.save("finalized_model.h5")