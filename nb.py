from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("data/train_val.csv")
pred_list = []

num_classes = 2
# split data for each class
attr = [
    "unnecessary",
    "mandatory",
    "pharma",
    "conspiracy",
    "political",
    "country",
    "rushed",
    "ingredients",
    "side-effect",
    "ineffective",
    "religious",
    "none",
]

accuracies = {}
training_accuracy = {}
precision = {}
recall = {}

df_unnecessary = pd.read_csv("data/df_unnecessary.csv")
df_mandatory = pd.read_csv("data/df_mandatory.csv")
df_pharma = pd.read_csv("data/df_pharma.csv")
df_conspiracy = pd.read_csv("data/df_conspiracy.csv")
df_political = pd.read_csv("data/df_political.csv")
df_country = pd.read_csv("data/df_country.csv")
df_rushed = pd.read_csv("data/df_rushed.csv")
df_ingredients = pd.read_csv("data/df_ingredients.csv")
df_side_effect = pd.read_csv("data/df_side_effect.csv")
df_ineffective = pd.read_csv("data/df_ineffective.csv")
df_religious = pd.read_csv("data/df_religious.csv")
df_none = pd.read_csv("data/df_none.csv")

df_all = [
    df_unnecessary,
    df_mandatory,
    df_pharma,
    df_conspiracy,
    df_political,
    df_country,
    df_rushed,
    df_ingredients,
    df_side_effect,
    df_ineffective,
    df_religious,
    df_none,
]

#shuffle each dataset in df_all
for i in range(len(df_all)):
    df_all[i] = df_all[i].sample(frac=1).reset_index(drop=True)
    
def ngram_vectorize(train_texts, train_labels, val_texts):
    # Vectorization parameters
    # Range (inclusive) of n-gram sizes for tokenizing text.
    NGRAM_RANGE = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 20000

    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = "word"

    # Minimum document/corpus frequency below which a token will be discarded.
    MIN_DOCUMENT_FREQUENCY = 2
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        "ngram_range": NGRAM_RANGE,  # Use 1-grams + 2-grams.
        "dtype": np.float64,
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": TOKEN_MODE,  # Split text into word tokens.
        "min_df": MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
    #x_pred = vectorizer.transform(pred_text)
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype("float32")
    x_val = selector.transform(x_val).astype("float32")
    return x_train, x_val

def get_accuracy(model, x_val, y_val):
    # Set the model to evaluation mode.
    model.eval()

    # Compute the predictions on the validation set.
    with torch.no_grad():
        logits = model(x_val)
        predictions = torch.argmax(logits, dim=1)

    # Compute the accuracy of the model on the validation set.
    correct_predictions = torch.eq(predictions, y_val).sum().item()
    total_predictions = y_val.shape[0]
    accuracy = correct_predictions / total_predictions

    return accuracy

from sklearn.metrics import recall_score

def get_recall(model, x_val, y_val):
    # Set the model to evaluation mode.
    model.eval()

    # Compute the predictions on the validation set.
    with torch.no_grad():
        logits = model(x_val)
        predictions = torch.argmax(logits, dim=1)

    # Compute the recall of the model on the validation set.
    recall = recall_score(y_val, predictions, average="macro")

    return recall

from sklearn.metrics import precision_score

def get_precision(model, x_val, y_val):
    # Set the model to evaluation mode.
    model.eval()

    # Compute the predictions on the validation set.
    with torch.no_grad():
        logits = model(x_val)
        predictions = torch.argmax(logits, dim=1)

    # Compute the precision of the model on the validation set.
    precision = precision_score(y_val, predictions, average="macro")

    return precision



def train_ngram_model(
    data,
    name,
    max_depth=None,
    random_state=None,
    # min_samples_split=2,
    # min_samples_leaf=1,
    # criterion = 'entropy'
    
):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)

    # Create model instance.
    try: 
        #load model from file 
        model = torch.load("models/" + name + ".pt")
    except:
        # Create model instance.
        model = MultinomialNB()

        # Fit the model.
        model.fit(x_train, train_labels)

        #save model 
        torch.save(model, "models/" + name + ".pt")
    # Print 5 sample tweets and their labels.
    print("\nSample tweets for class attribute:", name)
    sample_indices = np.random.choice(len(val_texts), 5, replace=False) # Select 5 random samples
    for idx in sample_indices:
        print("Tweet:", val_texts.iloc[idx])
        print("Label:", val_labels.iloc[idx])
    print("\n")

    # Compute accuracy.
    train_accuracy = accuracy_score(train_labels, model.predict(x_train))
    val_accuracy = accuracy_score(val_labels, model.predict(x_val))
    recall_1 = recall_score(val_labels, model.predict(x_val), average="macro")
    precision_1 = precision_score(val_labels, model.predict(x_val), average="macro")
    accuracies[name] = val_accuracy
    training_accuracy[name] = train_accuracy
    precision[name] = precision_1
    recall[name] = recall_1

    
    # Print the accuracy
    print("Training accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)
    
    # Print the recall
    print("recall", recall_1)
    
    # Print the precision
    print("precision", precision_1)

from sklearn.metrics import accuracy_score

def predict(model, train_texts, train_labels, val_texts):
    # Vectorizing the text data using your custom ngram_vectorize function
    x_val = ngram_vectorize(train_texts, train_labels, val_texts)[1]

    # Predicting the validation labels
    val_predictions = model.predict(x_val)

    # Calculating the accuracy of the model on the validation data
    return val_predictions


global pred_dict
pred_dict = {}
#for i in attr:
    #pred_dict[i] = []

#read test data
df_val = pd.read_csv("test_data/test.csv")

for ind, i in enumerate(df_all):
    #Split the data into training and testing sets
    #i, ind = df_all[0], 0
    print(attr[ind])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        i.iloc[:, 1], i.iloc[:, -1], test_size=0.2, random_state=42
    )
    data = (train_texts, train_labels), (val_texts, val_labels)

    train_ngram_model(data, name=attr[ind])

    #make predictions
    val_texts = df_val.iloc[:, 1]
    #load relevant model
    model = torch.load("models/" + attr[ind] + ".pt")
    pred_list.append(predict(model, train_texts, train_labels, val_texts))
print('\n')
print('Total attributes',attr)
print('\n')
print('Validation Accuracy',accuracies)
print('\n')
print('Training accuracy',training_accuracy)
print('\n')
print('Precision',precision)
print('\n')
print('Recall',recall)

#add predictions to dataframe

#save dataframe
# for j in range(len(pred_list[0])):
#     for i in range(len(pred_list)):
#         print(pred_list[i][j], end = " ")
#     print()