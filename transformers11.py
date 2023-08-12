from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import re
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv("data/train_val.csv")
texts = df['tweet']  # Replace with the name of your text column
labels = df['labels'] 
def clean(text):
    # Remove "@" tags
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove URLs
    text = re.sub(r'https?:\/\/\S+', '', text)
    
    # Remove special characters
    special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_char_pattern.sub('', text)

    return text

df['tweet'] = df['tweet'].apply(clean)
# Read your data
 # Replace with the name of your label column
from sklearn.preprocessing import LabelEncoder

# Encode the labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

# Tokenize the data using DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Create a Data Collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_labels = train_labels.astype(int)
val_labels = val_labels.astype(int)
unique_labels = np.unique(train_labels)
print(unique_labels)


def preprocess_function(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding='max_length',return_tensors='pt')

    # Debugging: Print shape and type of labels
    print("Labels shape:", np.shape(labels))
    print("Labels type:", type(labels[0]))

    return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': torch.tensor(labels)}

class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['labels'])

train_encodings = preprocess_function(train_texts.tolist(), train_labels.tolist())
val_encodings = preprocess_function(val_texts.tolist(), val_labels.tolist())

train_dataset = MyDataset(train_encodings)
val_dataset = MyDataset(val_encodings)
# Create a sequence classification model
num_classes = len(unique_labels)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define a function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': (predictions == labels).mean(),
        # Add other metrics if needed
    }

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
