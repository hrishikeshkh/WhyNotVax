from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Assume X_text and y_multilabel are the text data and corresponding multi-label targets
#Get the dataframe
df = pd.read_csv('data/super_clean.csv')
X_text = df['tweet']
#acess these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary
y_multilabel = df[['conspiracy','country','ineffective','ingredients','mandatory','none','pharma','political','religious','rushed','side-effect','unnecessary']]

# Split the data
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y_multilabel, random_state=42)

# Perform TF-IDF vectorization with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Select the top 20k features using f_classif
feature_selector = SelectKBest(chi2, k=min(20000, X_train_tfidf.shape[1]))
X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = feature_selector.transform(X_test_tfidf)

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(32, 64), # Example: Two hidden layers with 100 and 50 neurons
    activation='relu',
    solver='adam',
    alpha=0.3,
    learning_rate='constant',
    max_iter=10,
)

param_grid = {
    'base_estimator__hidden_layer_sizes': [(32, 64), (32, 64, 64)],
    'base_estimator__alpha' : [0.2, 0.5],
    #'base_estimator__learning_rate' : [0.0001, 0.0004],
    'base_estimator__max_iter' : [100, 500, 1000, 1500]
    # Add other hyperparameters as needed
}

# Create a Classifier Chain
chain = ClassifierChain(base_estimator=mlp_classifier, order='random', random_state=42)

# Fit the chain on the training data
#chain.fit(X_train_selected, y_train)


grid_search = GridSearchCV(chain, param_grid, cv=5, n_jobs= -1) # cv=5 specifies 5-fold cross-validation

# Fit the grid search to the data
grid_search.fit(X_train_selected, y_train)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

#save the best onto a file
with open('best_params.txt', 'w') as f:
    f.write(str(best_params))
    f.write('\n')
    f.write(str(best_estimator))

#also store it on a binary
import pickle
with open('best_params.bin', 'wb') as f:
    pickle.dump(best_params, f)
    pickle.dump(best_estimator, f)

y_pred = chain.predict(X_test_selected)

# Now, y_pred contains the multi-label predictions
print(y_pred[:5])

# Get the score
y_train_pred = chain.predict(X_train_selected)
print("training ", chain.score(X_train_selected, y_train))
print("testing ", chain.score(X_test_selected, y_test))