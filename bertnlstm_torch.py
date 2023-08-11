import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# Load and preprocess data
df = pd.read_csv('data/train_val_cleaned.csv')
tweets = df['tweet'].values
labels = df['labels'].values

# Use the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 100
tokenized_texts = [tokenizer.encode(tweet, add_special_tokens=True, max_length=max_length, truncation=True) for tweet in tweets]
padded_sequences = np.array(tokenized_texts)
attention_masks = (padded_sequences > 0).astype(int)

all_labels = set()
for label in labels:
    label_list = label.split()
    all_labels.update(label_list)

num_labels = len(all_labels)
label_mapping = {label: i for i, label in enumerate(all_labels)}
encoded_labels = np.zeros((len(labels), num_labels), dtype=np.int32)
for i, label in enumerate(labels):
    label_indices = [label_mapping[l] for l in label.split(' ')]
    encoded_labels[i, label_indices] = 1

# Split the data into train and test sets
X_train_ids, X_test_ids, y_train, y_test, X_train_masks, X_test_masks = train_test_split(
    padded_sequences, encoded_labels, attention_masks, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_ids = torch.tensor(X_train_ids, dtype=torch.long)
X_test_ids = torch.tensor(X_test_ids, dtype=torch.long)
X_train_masks = torch.tensor(X_train_masks, dtype=torch.long)
X_test_masks = torch.tensor(X_test_masks, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_ids, X_train_masks, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the integrated BERT-LSTM model
class BertLstmModel(nn.Module):
    def __init__(self, num_labels):
        super(BertLstmModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.dropout = nn.Dropout(0.6)
        self.conv1d = nn.Conv1d(128, 50, kernel_size=20, padding=0, stride=2)
        self.fc = nn.Linear(50, num_labels)

    def forward(self, input_ids, attention_mask):
        _, bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output)
        lstm_out = lstm_out[:, -1, :]  # Use the final time step's output
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)  # Conv1D expects channels last
        conv_out = self.conv1d(lstm_out)
        conv_out = conv_out.transpose(1, 2)  # Bring back to channels first
        logits = self.fc(conv_out)
        return torch.sigmoid(logits)

# Instantiate the model
model = BertLstmModel(num_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_ids, batch_masks, batch_labels in train_loader:
        batch_ids, batch_masks, batch_labels = batch_ids.to(device), batch_masks.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_ids, batch_masks)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_ids, test_masks = X_test_ids.to(device), X_test_masks.to(device)
    y_pred = model(test_ids, test_masks)
    test_loss = criterion(y_pred, y_test.to(device))
    accuracy = ((y_pred > 0.5) == y_test.to(device)).float().mean()

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
