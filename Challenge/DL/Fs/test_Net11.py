import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import hdf5plugin 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
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
    testdata = f["test/data"][:]
    testlabels = f["test/label"][:]

print(realdata.shape)
print(real_label.shape)
print(testdata.shape)
print(testlabels.shape)

traindata = realdata[:2250*128]
validdata = realdata[2250*128:]
trainlabels = real_label[:2250*128]
validlabels = real_label[2250*128:]

# for ldr a

traindata = realdata[:2250*128]
trainlabels = real_label[:2250*128]

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
# print(len(POI_lst))


err_lst = []

test_data = np.zeros([len(testdata), len(POI_lst)])
test_label = np.zeros([len(testdata), 1])
for i in range(len(testdata)):
    for j in range(len(POI_lst)):
        test_data[i][j] = testdata[i][POI_lst[j]]
    test_label[i] = int(testlabels[i][2]) + 1
    if test_label[i] in [1,2]:
        test_label[i] = 1


# Load the trained model
model = Net(len(POI_lst), 2)
model.load_state_dict(torch.load(f'Net11.pth'))

new_curves_tensor = torch.tensor(test_data, dtype=torch.float32)
model.eval()

with torch.no_grad():
    outputs = model(new_curves_tensor)
    _, predicted_labels = torch.max(outputs, dim=1)

sr = 0
total = 0
for i in range(len(test_label)):
    total += 1
    if int(predicted_labels[i].item()) != int(test_label[i].item()):
        print(testlabels[i], int(predicted_labels[i].item()), int(test_label[i].item()))
        err_lst.append(testlabels[i][0])
    else:
        sr += 1
print(f"[Net11 for ldr a] accuracy of model on test data = {sr/total*100}%")

err_lst = list(set(err_lst))
print("error list: ", len(err_lst), err_lst)
