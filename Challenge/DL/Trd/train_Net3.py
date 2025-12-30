import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import hdf5plugin, h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utils import *

with h5py.File("data_trdntt.h5", 'r') as f:
    data = f["train/data"][:]
    label = f["train/label"][:]
    testdata = f["test/data"][:]
    testlabel = f["test/label"][:]


cnt = 0
for i in range(len(label)):
    if label[i][2] == 13:
        cnt += 1
print(cnt)

cnt = [0,0]
for i in range(len(label)):
    if label[i][2] == 13:
        if label[i][0] < 2250:
            cnt[0] += 1
        else:
            cnt[1] += 1
    
print(cnt)
traindata = np.zeros([40385, 1000])
trainlabel = np.zeros([40385, 12])
validatedata = np.zeros([13522, 1000])
validatelabel = np.zeros([13522, 12])

cnt = [0,0]
for i in range(len(data)):
    if label[i][2] == 13:
        if label[i][0] < 2250:
            traindata[cnt[0]] = data[i]
            trainlabel[cnt[0]] = label[i]
            cnt[0] += 1
        else:
            validatedata[cnt[1]] = data[i]
            validatelabel[cnt[1]] = label[i]
            cnt[1] += 1
print(cnt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")


tbox_lst = [[0, 1], [0, 9], [0, 20]]
POI_lst = []
for k in range(len(tbox_lst)):
    hws = np.array([[tbox_lst[k][int(trainlabel[i][3])] for i in range(len(traindata))]]).transpose()
    POI = POI_selection_via_CPA(traindata, hws, False, 0.025)
    POI_lst += POI.tolist()
POI_lst = sorted(list(set(POI_lst)))
print(len(POI_lst), POI_lst)


train_data = np.zeros([len(traindata), len(POI_lst)])
validate_data = np.zeros([len(validatedata), len(POI_lst)])
train_labels = np.zeros([len(traindata), 1])
validate_labels = np.zeros([len(validatedata), 1])
for i in range(len(traindata)):
    for j in range(len(POI_lst)):
        train_data[i][j] = traindata[i][POI_lst[j]]
    train_labels[i] = int(trainlabel[i][3])
for i in range(len(validatedata)):
    for j in range(len(POI_lst)):
        validate_data[i][j] = validatedata[i][POI_lst[j]]
    validate_labels[i] = int(validatelabel[i][3])


train_dataset = Dataset(train_data, train_labels)
test_dataset = Dataset(validate_data, validate_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)

model = Net(len(POI_lst), 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.0001, weight_decay=1e-3, eps=1e-8)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

epochs = 300
best_loss = 1e10
train_losses = []
test_losses = []
for epoch in trange(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        accuracy = correct / total
    if best_loss > test_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), f'Net3.pth')

    scheduler.step()
