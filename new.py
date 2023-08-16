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
import nltk
nltk.download('stopwords')


# Hyperparameters for MLP
hyperparameters_per_class = {
    "conspiracy": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "country": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "ineffective": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "ingredients": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "mandatory": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "none": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "pharma": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "political": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "religious": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "rushed": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "side-effect": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha":0.3,
        "learning_rate": "constant",
        "max_iter": 100,
    },
    "unnecessary": {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "constant",
        "alpha": 0.3,
        "max_iter": 100,
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

    individual_classifiers.append(mlp_classifier)

print("Training complete for individual classifiers.")

# You can now use the individual classifiers to make predictions and combine them as needed

#save individual classifiers to a file
file = open('individual_classifiers.pkl', 'wb')
pickle.dump(individual_classifiers, file)



# Load the classifiers
with open('individual_classifiers.pkl', 'rb') as file:
    classifiers = pickle.load(file)

# Determine the number of classes

# Assuming X_test_selected contains the test features

# Initialize y_pred with zeros
y_pred = np.zeros((X_test_selected.shape[0], len(classifiers)))

# Loop through the classifiers
for idx, classifier in enumerate(classifiers):
    # Make predictions for each row in X_test_selected
    predictions = classifier.predict(X_test_selected)
    
    # Store the predictions in y_pred for the corresponding classifier
    y_pred[:, idx] = predictions

# Now y_pred contains the predictions for each classifier
print(classification_report(y_pred, y_test))  # Assuming y_test is properly prepared
