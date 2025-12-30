import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import torch
import hdf5plugin, h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm, trange
from utils import *

with h5py.File("data_secntt.h5", 'r') as f:
    realdata = f["train/data"][:]
    real_label = f["train/label"][:]

print(len(realdata), len(realdata[0]))
cnt = [0,0,0]
cnt0 = [0,0,0]
for i in range(len(realdata)):
    if real_label[i][0] < 2250:
        cnt[real_label[i][2]] += 1
    else:
        cnt0[real_label[i][2]] += 1
print(cnt, cnt0)

traindata0 = np.zeros([10083, 1000])
trainlabel0 = np.zeros([10083, 8])
traindata1 = np.zeros([112568, 1000])
trainlabel1 = np.zeros([112568, 8])
traindata2 = np.zeros([10225, 1000])
trainlabel2 = np.zeros([10225, 8])

validatedata0 = np.zeros([3423, 1000])
validatelabel0 = np.zeros([3423, 8])
validatedata1 = np.zeros([37574, 1000])
validatelabel1 = np.zeros([37574, 8])
validatedata2 = np.zeros([3373, 1000])
validatelabel2 = np.zeros([3373, 8])

traindata = [traindata0, traindata1, traindata2]
trainlabel = [trainlabel0, trainlabel1, trainlabel2]
validatedata = [validatedata0, validatedata1, validatedata2]
validatelabel = [validatelabel0, validatelabel1, validatelabel2]

train_cnt = [0,0,0]
validate_cnt = [0,0,0]
for i in range(len(realdata)):
    idx = real_label[i][2]
    if real_label[i][0] < 2250:
        traindata[idx][train_cnt[idx]] = realdata[i]
        trainlabel[idx][train_cnt[idx]] = real_label[i]
        train_cnt[idx] += 1
    else:
        validatedata[idx][validate_cnt[idx]] = realdata[i]
        validatelabel[idx][validate_cnt[idx]] = real_label[i]
        validate_cnt[idx] += 1

print(train_cnt, validate_cnt)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")

for NNno in [0, 2, 1]:
    tdata = traindata[NNno]
    tlabel = trainlabel[NNno]
    BOX = [
        [[14, 6], [21, 11]],
        [[0, 1], [0, 16], [0, 14]],
        [[11, 12], [12, 25]],
    ]
    tbox_lst = BOX[NNno]
    POI_lst = []
    for k in range(len(tbox_lst)):
        t = [tbox_lst[k][int(tlabel[i][3])] for i in range(len(tdata))]
        POI = POI_selection_via_CPA(tdata, np.array([t]).transpose(), False, 0.05)
        POI_lst += POI.tolist()
    POI_lst = sorted(list(set(POI_lst)))
    print(len(POI_lst), POI_lst)

    vdata = validatedata[NNno]
    vlabel = validatelabel[NNno]

    train_data = np.zeros([len(tdata), len(POI_lst)])
    validate_data = np.zeros([len(vdata), len(POI_lst)])
    train_labels = np.zeros([len(tdata), 1])
    validate_labels = np.zeros([len(vdata), 1])
    for i in range(len(tdata)):
        for j in range(len(POI_lst)):
            train_data[i][j] = tdata[i][POI_lst[j]]
        train_labels[i] = int(tlabel[i][3])
    for i in range(len(vdata)):
        for j in range(len(POI_lst)):
            validate_data[i][j] = vdata[i][POI_lst[j]]
        validate_labels[i] = int(vlabel[i][3])

    train_dataset = Dataset(train_data, train_labels)
    test_dataset = Dataset(validate_data, validate_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = Net(len(POI_lst), 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.0001, weight_decay=1e-3, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    print("--Trainning START--")

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
            torch.save(model.state_dict(), f'Net2{NNno}.pth')

        scheduler.step()

    print("--Trainning END--")
