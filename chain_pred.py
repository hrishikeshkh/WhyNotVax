# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Assume X_text and y_multilabel are the text data and corresponding multi-label targets
# Get the dataframe
# %%
df = pd.read_csv("data/super_clean.csv")
X_text = df["tweet"]
# acess these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary
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

# %%

# Split the data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_multilabel, random_state=42
)

# Perform TF-IDF vectorization with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Select the top 20k features (if available)
feature_selector = SelectKBest(chi2, k=min(20000, X_train_tfidf.shape[1]))
X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = feature_selector.transform(X_test_tfidf)

# %%
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(32, 64), 
    activation="relu",
    solver="adam",
    alpha=0.5,
    learning_rate="constant",
    max_iter=10,
)

# Create a Classifier Chain
chain = ClassifierChain(base_estimator=mlp_classifier, order="random", random_state=42)

# %%

# Fit the chain on the training data
chain.fit(X_train_selected, y_train)

# %%

# Predict on the test data
y_pred = chain.predict(X_test_selected)

# Now, y_pred contains the multi-label predictions
print(y_pred[:5])

# Get the score
y_train_pred = chain.predict(X_train_selected)
print("training ", chain.score(X_train_selected, y_train))
print("testing ", chain.score(X_test_selected, y_test))

# %%

#save the model
import pickle
filename = 'finalized_model.sav'
pickle.dump(chain, open(filename, 'wb'))