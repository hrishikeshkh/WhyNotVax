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
from sklearn.model_selection import RandomizedSearchCV

# Assume X_text and y_multilabel are the text data and corresponding multi-label targets
#Get the dataframe
df = pd.read_csv('data/balanced_super_clean.csv')
X_text = df['tweet']
#acess these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary
y_multilabel = df[['conspiracy','country','ineffective','ingredients','mandatory','none','pharma','political','religious','rushed','side-effect','unnecessary']]

# Split the data
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y_multilabel, random_state=42)

df_test = pd.read_csv('data/super_clean.csv')
df_filtered = df_test[~df_test['tweet'].isin(X_train_text)]
X_test_text = df_filtered['tweet']
y_test = df_filtered[['conspiracy','country','ineffective','ingredients','mandatory','none','pharma','political','religious','rushed','side-effect','unnecessary']]

# Perform TF-IDF vectorization with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Select the top 20k features using f_classif
feature_selector = SelectKBest(chi2, k=min(2000, X_train_tfidf.shape[1]))
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

# param_grid = {
#     'base_estimator__hidden_layer_sizes': [(32, 32), (32, 64), (32, 64, 64), (32, 128, 64), (64, 128, 64), (64, 128, 128), (128, 128, 128)],
#     'base_estimator__alpha' : [0.1, 0.2, 0.5],
#     #'base_estimator__learning_rate' : [0.0001, 0.0004],
#     'base_estimator__max_iter' : [200, 500, 1000, 2000, 3000]
#     # Add other hyperparameters as needed
# }

param_grid = {
    'base_estimator__hidden_layer_sizes': [(32), (32, 32), (32, 64), (64, 64), (32, 32, 32), (32, 64, 32), (64, 64, 64), (32, 64, 64), (64, 64, 128), (64, 128, 128), (128, 128, 128)],
    'base_estimator__alpha' : [0.1, 0.2, 0.4, 0.5],
    #'base_estimator__learning_rate' : [0.0001, 0.0004],
    'base_estimator__max_iter' : [100, 200, 500, 1000, 2000]
    # Add other hyperparameters as needed
}

# Create a Classifier Chain
chain = ClassifierChain(base_estimator=mlp_classifier, order='random', random_state=42)

# Fit the chain on the training data
#chain.fit(X_train_selected, y_train)


random_search = RandomizedSearchCV(chain, param_grid, n_iter=15, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit the grid search to the data
random_search.fit(X_train_selected, y_train)

# Open the log file to write
# Get the best parameters and estimator
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

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

'''
y_pred = chain.predict(X_test_selected)

# Now, y_pred contains the multi-label predictions
print(y_pred[:5])

# Get the score
y_train_pred = chain.predict(X_train_selected)
print("training ", chain.score(X_train_selected, y_train))
print("testing ", chain.score(X_test_selected, y_test))

df = pd.read_csv('data/super_clean.csv')


def append_labels_and_drop_columns(df):
    labels = ["conspiracy", "country", "ineffective", "ingredients", "mandatory", "none", "pharma", "political", "religious", "rushed", "side-effect", "unnecessary"]

    # Function to find the labels corresponding to the one-hot encoded columns
    def get_labels(row):
        selected_labels = [label for label in labels if row[label] == 1]
        return ', '.join(selected_labels[:3]) # Limit to 3 labels

    # Apply the function to create the 'labels' column
    df['labels'] = df.apply(get_labels, axis=1)

    # Drop the one-hot encoded columns
    df.drop(labels, axis=1, inplace=True)

    return df


df = append_labels_and_drop_columns(df)
print(df)
csv_file_path = 'data/processed_data.csv'
df.to_csv(csv_file_path, index=False)'''