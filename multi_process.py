import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Generate some dummy data
X = torch.randn(10000, 10)
y = torch.randint(0, 2, (10000,))

# Define your dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define your model and optimizer
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define a function to train the model on a batch
def train_batch(batch):
    inputs, labels = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()

# Use multiprocessing to parallelize the training
with Pool(processes=4) as pool:
    for epoch in range(10):
        pool.map(train_batch, dataloader)
        \


#{'unnecessary': 0.9289672544080605, 'mandatory': 0.9385390428211587, 'pharma': 0.8876574307304785, 'conspiracy': 0.9486146095717884, 'political': 0.945088161209068, 'country': 0.981360201511335, 'rushed': 0.9007556675062972, 'ingredients': 0.9561712846347608, 'side-effect': 0.8382871536523929, 'ineffective': 0.8670025188916877, 'religious': 0.9944584382871536, 'none': 0.9445843828715366}

