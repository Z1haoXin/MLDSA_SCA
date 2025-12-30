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
from utils import *

with h5py.File("data_trdntt.h5", 'r') as f:
    data = f["train/data"][:]
    label = f["train/label"][:]
    testdata = f["test/data"][:]
    testlabel = f["test/label"][:]

traindata = np.zeros([40385, 1000])
trainlabel = np.zeros([40385, 12])
cnt = 0
for i in range(len(data)):
    if label[i][2] == 13:
        if label[i][0] < 2250:
            traindata[cnt] = data[i]
            trainlabel[cnt] = label[i]
            cnt += 1
print(cnt)

err_lst = []

tbox_lst = [[0,1],[0,9],[0,20]]
POI_lst = []
for k in range(len(tbox_lst)):
    hws = np.array([[tbox_lst[k][int(trainlabel[i][3])] for i in range(len(traindata))]]).transpose()
    POI = POI_selection_via_CPA(traindata, hws, False, 0.025)
    POI_lst += POI.tolist()
POI_lst = sorted(list(set(POI_lst)))
print(len(POI_lst), POI_lst)


test_data = np.zeros([len(testdata), len(POI_lst)])
test_labels = np.zeros([len(testdata), 1])
for i in range(len(testdata)):
    for j in range(len(POI_lst)):
        test_data[i][j] = testdata[i][POI_lst[j]]
    test_labels[i] = int(testlabel[i][3])


model = Net(len(POI_lst), 2)
model.load_state_dict(torch.load(f'Net3.pth'))

new_curves_tensor = torch.tensor(test_data, dtype=torch.float32)
model.eval()

with torch.no_grad():
    outputs = model(new_curves_tensor)
    _, predicted_labels = torch.max(outputs, dim=1)

sr = 0
total = 0
for i in range(len(test_labels)):
    if testlabel[i][2] == 13:
        total += 1
        if int(predicted_labels[i].item()) != int(test_labels[i].item()):
            print(testlabel[i], int(predicted_labels[i].item()), int(test_labels[i].item()))
            err_lst.append(testlabel[i][0])
        else:
            sr += 1
print(f"[ Net3 ] total = {total}, Accuracy of Net3 on test data = {sr/total*100}%")
err_lst = list(set(err_lst))
print("error list: ", len(err_lst), err_lst)
