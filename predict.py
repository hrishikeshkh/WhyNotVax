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
#import naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
#import SVM
from sklearn import svm
#imoport random forest
from sklearn.ensemble import RandomForestClassifier
#import knn
from sklearn.neighbors import KNeighborsClassifier
#import gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords

#laod the model 
import pickle
filename = 'finalized_model.sav'
chain = pickle.load(open(filename, 'rb'))

df_val =pd.read_csv('test_data/test.csv')
X_text_val = df_val["tweet"]
# acess these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary

#srop the stop words from X_text
stop = stopwords.words('english')
X_text_val = X_text_val.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

final_domain_specific_stop_words = [
    "covid", "covid19", "vaccines", "virus", "people", "dont", "im", "just", "want", "need",
    "inoculation", "immunization", "dose", "shot", "trial", "effect", "health", "risk", "trust",
    "government", "rate", "death"
]

X_text_val = X_text_val.apply(lambda x: ' '.join([word for word in x.split() if word not in (final_domain_specific_stop_words)]))

# acess these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary

# %%
df = pd.read_csv("data/super_clean.csv")
X_text = df["tweet"]
# access these columns conspiracy,country,ineffective,ingredients,mandatory,none,pharma,political,religious,rushed,side-effect,unnecessary
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

#srop the stop words from X_text
stop = stopwords.words('english')
X_text = X_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

final_domain_specific_stop_words = [
    "covid", "covid19", "vaccines", "virus", "people", "dont", "im", "just", "want", "need",
    "inoculation", "immunization", "dose", "shot", "trial", "effect", "health", "risk", "trust",
    "government", "rate", "death"
]

X_text = X_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (final_domain_specific_stop_words)]))


# Split the data
'''
df_test = pd.read_csv('data/super_clean.csv')
df_filtered = df_test[~df_test['tweet'].isin(X_train_text)]
X_test_text = df_filtered['tweet']
y_test = df_filtered[['conspiracy','country','ineffective','ingredients','mandatory','none','pharma','political','religious','rushed','side-effect','unnecessary']]
'''



# Perform TF-IDF vectorization with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_text)
X_test_val_tfidf = tfidf_vectorizer.transform(X_text_val)

# Select the top 20k features (if available)
#for i in range(2000, 20000, 2000):
feature_selector = SelectKBest(chi2, k=min(20000, X_train_tfidf.shape[1]))
X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_multilabel)
X_test_selected = feature_selector.transform(X_test_val_tfidf)

y_pred = chain.predict(X_test_selected)

# Now, y_pred contains the multi-label predictions
print(y_pred[:5])

def append_predictions_and_save(csv_path, predictions, labels, new_file_path):
    # Load the original CSV file
    data = csv_path

    # Convert the predictions to a DataFrame and set the column names as the labels
    predictions_df = pd.DataFrame(predictions, columns=labels)

    # Concatenate the original data with the predictions
    final_data = pd.concat([data, predictions_df], axis=1)

    # Save the final DataFrame to the new CSV file
    final_data.to_csv(new_file_path, index=False)
    print(f"File saved to {new_file_path}")

    return final_data

new_file_path = 'data/test_with_1hot.csv'
predictions = y_pred
labels = ["conspiracy", "country", "ineffective", "ingredients", "mandatory", "none", "pharma", "political", "religious", "rushed", "side-effect", "unnecessary"]
final_data = append_predictions_and_save(df_val, predictions, labels, new_file_path)

def append_labels_and_drop_columns(df):
    labels = ["conspiracy", "country", "ineffective", "ingredients", "mandatory", "none", "pharma", "political", "religious", "rushed", "side-effect", "unnecessary"]

    # Function to find the labels corresponding to the one-hot encoded columns
    def get_labels(row):
        selected_labels = [label for label in labels if row[label] == 1]
        return ' '.join(selected_labels[:3]) # Limit to 3 labels

    # Apply the function to create the 'labels' column
    df['labels'] = df.apply(get_labels, axis=1)

    # Drop the one-hot encoded columns
    df.drop(labels, axis=1, inplace=True)

    return df

df_label =pd.read_csv('data/test_with_1hot.csv')
df_label = append_labels_and_drop_columns(df_label)
print(df_label)
csv_file_path = 'data/processed_data.csv'
df_label.to_csv(csv_file_path, index=False)
# Get the score
# y_train_pred = chain.predict(X_train_selected)
# print("training with "," features", chain.score(X_train_selected, y_multilabel))
# print("testing with", " features ", chain.score(X_test_selected, y_pred))
