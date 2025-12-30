import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm, trange
from utils import *

with h5py.File("data_fsntt.h5", 'r') as f:
    realdata = f["train/data"][:]
    real_label = f["train/label"][:]

traindata = realdata[:2250*128]
validdata = realdata[2250*128:]
trainlabels = real_label[:2250*128]
validlabels = real_label[2250*128:]

print("### Train Net11 ###")
BOX = [
    [[32,0,1]],
]
tbox_lst = BOX[0]
POI_lst = []
for k in range(len(tbox_lst)):
    hws = np.array([[tbox_lst[k][int(trainlabels[i][2])+1] for i in range(len(trainlabels))]]).transpose()
    POI = POI_selection_via_CPA(traindata, hws, False, 0.035)
    POI_lst += POI.tolist()
POI_lst = sorted(list(set(POI_lst)))
print(len(POI_lst), POI_lst)


train_data = np.zeros([len(traindata), len(POI_lst)])
train_labels = np.zeros([len(traindata), 1])
for i in range(len(traindata)):
    for j in range(len(POI_lst)):
        train_data[i][j] = traindata[i][POI_lst[j]]
    train_labels[i] = int(trainlabels[i][2]) + 1
    if train_labels[i] in [1, 2]:
        train_labels[i] = 1

valid_data = np.zeros([len(validdata), len(POI_lst)])
valid_labels = np.zeros([len(validdata), 1])
for i in range(len(validdata)):
    for j in range(len(POI_lst)):
        valid_data[i][j] = validdata[i][POI_lst[j]]
    valid_labels[i] = int(validlabels[i][2]) + 1
    if valid_labels[i] in [1, 2]:
        valid_labels[i] = 1

train_dataset = Dataset(train_data, train_labels)
test_dataset = Dataset(valid_data, valid_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

model = Net(len(POI_lst), 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.0001, weight_decay=1e-3, eps=1e-8)
scheduler = CosineAnnealingLR(optimizer, T_max=10) 

epochs = 300
best_loss = 1
train_losses = []
test_losses = []
for epoch in trange(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        accuracy = correct / total
    scheduler.step()
    if best_loss > test_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'Net11.pth')
    

plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()


