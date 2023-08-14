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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV


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
    
def ngram_vectorize(train_texts, train_labels, val_texts, name):
    try:
        # Try to load the saved vectorizer and selector
        loaded_objects = torch.load("vectorizers/" + name + "_vectorizer.pt")
        vectorizer = loaded_objects['vectorizer']
        selector = loaded_objects['selector']
    except:
        # Vectorization parameters
        NGRAM_RANGE = (1, 2)
        TOP_K = 20000
        TOKEN_MODE = "word"
        MIN_DOCUMENT_FREQUENCY = 2
        kwargs = {
            "ngram_range": NGRAM_RANGE,
            "dtype": np.float64,
            "strip_accents": "unicode",
            "decode_error": "replace",
            "analyzer": TOKEN_MODE,
            "min_df": MIN_DOCUMENT_FREQUENCY,
        }
        vectorizer = TfidfVectorizer(**kwargs)

        # Learn vocabulary from training texts and vectorize training texts.
        x_train = vectorizer.fit_transform(train_texts)

        # Select top 'k' of the vectorized features.
        selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
        selector.fit(x_train, train_labels)

        # Save the vectorizer and selector
        save_objects = {'vectorizer': vectorizer, 'selector': selector}
        torch.save(save_objects, "vectorizers/" + name + "_vectorizer.pt")
    else:
        # If the vectorizer and selector are loaded, transform the training texts
        x_train = vectorizer.transform(train_texts)
        x_train = selector.transform(x_train).astype("float32")

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
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

def assign_classes(df, pred_list):
    pred_list_transposed = list(zip(*pred_list))
    
    print("Length of pred_list:", len(pred_list))
    print("Length of pred_list_transposed:", len(pred_list_transposed))
    print("Number of rows in df_val:", len(df))
    print("Sample of pred_list:", pred_list[:1])
    print("Sample of df_val:", df.head())

    if len(pred_list_transposed) != len(df):
        raise ValueError("Length of pred_list must match the number of rows in the dataframe.")

    classes = [
        "unnecessary", "mandatory", "pharma", "conspiracy", "political",
        "country", "rushed", "ingredients", "side_effect", "ineffective",
        "religious", "none"
    ]

    def get_class_name(row_pred):
        return ', '.join([class_name for idx, class_name in enumerate(classes) if row_pred[idx] == 1])

    predicted_classes = [get_class_name(row) for row in pred_list_transposed]

    result_df = pd.DataFrame({
        'tweet': df['tweet'].iloc[:], # Adjusting to match the length
        'predicted_class': predicted_classes
    })

    return result_df


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
    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts, name)
    print("Training shape:", x_train.shape)
    print("Validation shape:", x_val.shape)

    # Create model instance.
    try: 
        model = torch.load("models/" + name + "_model.pt")
    except:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, min_samples_split=2, min_samples_leaf=1, criterion = 'entropy')

        # Print 5 sample tweets and their labels.
        model.fit(x_train, train_labels)

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

    return model

from sklearn.metrics import accuracy_score

def predict(model, train_texts, train_labels, val_texts, name):
    # Vectorizing the text data using your custom ngram_vectorize function
    #set model to eval
    model.eval()
    x_val = ngram_vectorize(train_texts, train_labels, val_texts, name)[1]

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
    pred_list.append(predict(model, train_texts, train_labels, val_texts, attr[ind]))
    
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
result_df = assign_classes(df_val, pred_list)

#You can then save or print the result_df as needed
print(result_df.head())
result_df.to_csv("predictions_with_classes.csv", index=False)
#add predictions to dataframe

#save dataframe
for j in range(len(pred_list[0])):
    for i in range(len(pred_list)):
        print(pred_list[i][j], end = " ")
    print()
