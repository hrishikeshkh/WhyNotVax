# %% [markdown]
# SC making the NN a pytorch model

# %%
print("Sc Vax Classify")

# %%
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

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from imblearn.over_sampling import SMOTE


# %%
# get the data
df = pd.read_csv("data/train_val.csv")
df.head()

# %%
# std defn
num_classes = 2

# %%
import pandas as pd

# Read the data
data = pd.read_csv('data/train_val.csv')

df_unnecessary = data.copy()
df_unnecessary["unnecessary"] = df_unnecessary["labels"].apply(
    lambda x: 1 if "unnecessary" in x else 0
)

# Filter rows where 'unnesecary' column is 1
data_fil = df_unnecessary[df_unnecessary['unnecessary'] == 1]

#repeat the rows 10 times
data_fil = data_fil.loc[data_fil.index.repeat(5)]

# Add the new rows to the original data
df_unnecessary = pd.concat([df_unnecessary, data_fil])


df_unnecessary = df_unnecessary.drop(columns=['labels'])

print(df_unnecessary.head())

#print the number of 1s in the column unnsecessary
print(df_unnecessary['unnecessary'].value_counts())
# Save the new data
df_unnecessary.to_csv('data/df_unnecessary.csv', index=False)

# %%
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
# split data for each class

# copy the df and modify such that for those rows where the string value of labels column contains 'unnecessary', set the label columnn to 1 else 0
# make a copy of the df and set it to another variable
# df_unnecessary = df.copy()
# df_unnecessary["unnecessary"] = df_unnecessary["labels"].apply(
#     lambda x: 1 if "unnecessary" in x else 0
# )
# copy the df and modify such that for those rows where the string value of labels column contains 'mandatory', set the label columnn to 1 else 0

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

# %%
def ngram_vectorize(train_texts, train_labels, val_texts):
    # Vectorization parameters
    # Range (inclusive) of n-gram sizes for tokenizing text.
    NGRAM_RANGE = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 2000

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

# %%
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

# %%
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

# %%
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

# %%
# pytorch nn model
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers, units, dropout_rate, input_shape, num_classes):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(input_shape, units)
        self.relu1 = nn.ReLU()
        self.layers = nn.ModuleList()
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(units, units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Linear(units, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# %% [markdown]
# we have a bunch of options in front of us: binary chain, ensembling (best but maybe complex to choose b/w ensembling methods), power set (not rec)

# %%
def train_ngram_model(
    data,
    name,
    learning_rate=10e-3,
    epochs=100,
    batch_size=128,
    layers=2,
    units=64,
    dropout_rate=0.02,
):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)
    #print(x_train.shape)
    x_train_int = int(x_train.shape[0]), int(x_train.shape[1])
    #Create model instance.
    model = MLP(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train_int[1],
        num_classes=2,
    )

    
    # Compile model with learning parameters.
    if num_classes == 2:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()


    x_train = torch.from_numpy(x_train.toarray()).float()
    train_tuple = torch.tensor(train_labels.values)
    y_train = F.one_hot(train_tuple, num_classes=2).float()  

    # Create a TensorDataset from your training data.
    train_dataset = TensorDataset(x_train, y_train)

    # Create a DataLoader from the TensorDataset.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

    num_epochs = epochs

    model = nn.DataParallel(model)
    device = torch.device("cpu")
    model.to(device)

    try:
        model.load_state_dict(torch.load("models/model" + name + ".pt"))
    except:
        for epoch in range(num_epochs):
            model.train()  # Set the model in training mode
            total_step = len(train_loader)
            
            for i, (batch_data, batch_labels) in enumerate(train_loader):
                # Move batch data and labels to the specified device
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print or log training progress
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


    first_row = x_val.toarray()[0]

    x_val_tensor = torch.from_numpy(x_val.toarray()).float()
    y_val_tensor = torch.tensor(val_labels.values)
    accuracy = get_accuracy(model, x_val_tensor, y_val_tensor)

    # Print the accuracy.
    print("Validation accuracy:", accuracy)
    accuracies[name] = accuracy
    torch.save(model.state_dict(), "models/model" + name + ".pt")

    #print the recall
    print("recall", get_recall(model, x_val_tensor, y_val_tensor))
    
    #print the precision
    print("precision", get_precision(model, x_val_tensor, y_val_tensor))
    
    #print the training accuracry
    print("training accuracy", get_accuracy(model, x_train, train_tuple))
    

    first_row_tensor = torch.tensor(first_row)

    train_labels_list = train_labels.tolist()

   

    # model.eval()
    # with torch.no_grad():
    #     prediction = model.forward(torch.from_numpy(x_pred.toarray()))


    

# %% [markdown]
# 

# %%
for ind, i in enumerate(df_all):
    # Split the data into training and testing sets
    #i, ind = df_all[0], 0
    print(attr[ind])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        i.iloc[:, 1], i.iloc[:, -1], test_size=0.2, random_state=42
    )
    data = (train_texts, train_labels), (val_texts, val_labels)

    train_ngram_model(data, name=attr[ind])

# %%
print(accuracies)
print(attr)

# %% [markdown]
# {'unnecessary': 0.9289672544080605, 'mandatory': 0.9385390428211587, 'pharma': 0.8876574307304785, 'conspiracy': 0.9486146095717884, 'political': 0.945088161209068, 'country': 0.981360201511335, 'rushed': 0.9007556675062972, 'ingredients': 0.9561712846347608, 'side-effect': 0.8382871536523929, 'ineffective': 0.8670025188916877, 'religious': 0.9944584382871536, 'none': 0.9445843828715366}
# 

# %%
import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 7))
x = range(len(accuracies))
X = list(accuracies.keys())
y = list(accuracies.values())
plt.plot(X, y, marker='o')
plt.xticks(x, X, rotation="vertical")
plt.ylim([0, 1])
for i in range(len(x)):
    plt.text(x[i], y[i]-0.02, f'{y[i]:.4f}', ha='center', va='top', rotation='vertical')
plt.show()
plt.bar(X, y, color='red')
plt.xticks(x, X, rotation="vertical")
for i in range(len(x)):
    plt.text(x[i], y[i]-0.02, f'{y[i]:.4f}', ha='center', va='top', rotation='vertical')
plt.show()

# %% [markdown]
# TODO: make traning parallel
# TODO: account for data imbalance
# TODO: ramp up epochs to max and take top 20k features again (change the df_min)