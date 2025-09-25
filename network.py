#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.transforms import v2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 318
hidden_size = 128
num_classes = 2
num_epochs = 100
batch_size = 32
learning_rate = 0.001

class LolDataset(Dataset):
    def __init__(self, df, target_column, transform=None):
        self.data = df
        self.data = pd.get_dummies(self.data, dtype=float)
        self.transform = transform
        self.target_column = target_column

        self.X = self.data.drop(columns=[target_column]).values.astype(float)
        self.y = self.data[target_column].values

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
        return x
    
    @property
    def classes(self):
        return self.data.classes
    
transform = v2.Compose([
    transforms.ToTensor()
])

df = pd.read_csv("predict_train.csv")
df = df.drop(columns="Unnamed: 0")
#results = df.pop('result')
#normalized_df = (df-df.mean())/df.std()
#normalized_df.insert(0, "result", results)

#print(normalized_df.head(10).to_string())

#%%

split_idx = int(0.7 * len(df))

df_train = df[:split_idx]
df_test = df[split_idx:]

dataset = LolDataset(df_train, "result", transform)
testset = LolDataset(df_test, "result", transform)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

#%%
for matches, label, in data_loader:
    print(matches.shape)
    break

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
       super(BinaryClassifier, self).__init__()
       self.l1 = nn.Linear(input_size, hidden_size)
       self.relu = nn.ReLU()
       self.relu2 = nn.ReLU()
       self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.relu2(out)
        out = self.l2(out)

        return out
    
model = BinaryClassifier(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.1,1.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(data_loader)
for epoch in range(num_epochs):
    for i, (matches, labels) in enumerate(data_loader):
        matches = matches.to(device)
        labels = labels.long().to(device)

        outputs = model(matches)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}]/{num_epochs}], Step [{i+1}]/{n_total_steps}], Loss: {loss.item():.4f}')

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)
    print(n_samples)
    for matches, labels in test_loader:
        matches = matches.to(device)
        labels = labels.to(device)

        outputs = model(matches)

        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()
        preds = torch.argmax(outputs, dim=1)   # probability of "win"
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = n_correct / n_samples

print(f'Accuracy on {n_samples} matches: {100 * acc}%')

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["loss", "win"])
disp.plot(cmap=plt.cm.Blues)
plt.show()
print(classification_report(all_labels, all_preds, target_names=["Loss", "Win"]))

