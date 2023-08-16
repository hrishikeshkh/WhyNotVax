from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle


# Hyperparameters for MLP
hyperparameters_per_class = {
    "conspiracy": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "country": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "ineffective": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "ingredients": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "mandatory": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "none": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "pharma": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "political": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "religious": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.1,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "rushed": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "side-effect": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha":1,
        "learning_rate": "constant",
        "max_iter": 5,
    },
    "unnecessary": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "constant",
        "alpha": 0.1,
        "max_iter": 5,
    },# Add more class-specific hyperparameters here
}


# Splitting the dataset into features and labels
df = pd.read_csv("data/super_clean.csv")
X = df["tweet"]
y = df[ [
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
    ]]

stop = stopwords.words('english')
X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

final_domain_specific_stop_words = [
    "covid", "covid19", "vaccines", "virus", "people", "dont", "im", "just", "want", "need",
    "inoculation", "immunization", "dose", "shot", "trial", "effect", "health", "risk", "trust",
    "government", "rate", "death"
]

X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (final_domain_specific_stop_words)]))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
# List to store the individual classifiers
individual_classifiers = []

# Iterating through each class and training an individual classifier
for class_name in y_train.columns:
    print(f"Training classifier for class: {class_name}")
    
    # Creating the classifier with the defined hyperparameters
    hyperparameters = hyperparameters_per_class.get(class_name, {})
    mlp_classifier = MLPClassifier(**hyperparameters, verbose= True, random_state=42)
    
    # Extracting the labels for the current class
    y_train_class = y_train[class_name]

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Select the top 20k features (if available)
    #for i in range(2000, 20000, 2000):
    feature_selector = SelectKBest(f_classif, k=min(15000, X_train_tfidf.shape[1]))
    X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train_class)
    X_test_selected = feature_selector.transform(X_test_tfidf)
        
    # Fitting the classifier
    mlp_classifier.fit(X_train_selected, y_train_class)
    # Storing the individual classifier
    #print accuracy
    y_pred = mlp_classifier.predict(X_test_selected)
    print("Accuracy for : ", class_name ,accuracy_score(y_test[class_name], y_pred))

    #print classification report
    print(classification_report(y_test[class_name], y_pred))
    individual_classifiers.append(mlp_classifier)

print("Training complete for individual classifiers.")

# You can now use the individual classifiers to make predictions and combine them as needed

#save individual classifiers to a file
file = open('individual_classifiers.pkl', 'wb')
pickle.dump(individual_classifiers, file)



'''
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
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_multilabel, random_state=42
)

'''
'''
df_test = pd.read_csv('data/super_clean.csv')
df_filtered = df_test[~df_test['tweet'].isin(X_train_text)]
X_test_text = df_filtered['tweet']
y_test = df_filtered[['conspiracy','country','ineffective','ingredients','mandatory','none','pharma','political','religious','rushed','side-effect','unnecessary']]
'''
'''



# Perform TF-IDF vectorization with unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Select the top 20k features (if available)
#for i in range(2000, 20000, 2000):
feature_selector = SelectKBest(chi2, k=min(15000, X_train_tfidf.shape[1]))
X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = feature_selector.transform(X_test_tfidf)

# %%
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(64,64,64), 
    activation="relu",
    solver="adam",
    alpha=0.1,
    learning_rate="adaptive",
    max_iter=200,
)

nb_classifier = MultinomialNB()
#svm = svm.SVC()
#random_forest = RandomForestClassifier(n_estimators=5000, max_depth=30, random_state=42)
#knn = KNeighborsClassifier(n_neighbors=50)
#gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)

# Create a Classifier Chain
chain = ClassifierChain(base_estimator=mlp_classifier, order="random", random_state=42, verbose=True)

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
print("training with "," features", chain.score(X_train_selected, y_train))
print("testing with", " features ", chain.score(X_test_selected, y_test))

# %%

#save the model
import pickle
filename = 'finalized_model.sav'
pickle.dump(chain, open(filename, 'wb'))

'''