import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import hdf5plugin 
import h5py
import numpy as np
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange

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

# select POI for BF(a, b) using CPA
# tbox_lst = [
#     [32,0,1], # ldr b
#     [23, 0, 10,], # zeta*b
#     [14, 0, 19], # t
# ]
# POI_lst = []
# for k in range(3):
#     hws = np.array([[tbox_lst[k][int(trainlabels[i][3])+1] for i in range(len(traindata))]]).transpose()
#     POI = POI_selection_via_CPA(traindata, hws, True, 0.1)
#     POI_lst += POI.tolist()
# POI_lst = sorted(list(set(POI_lst)))

# del first 20-100 points, not leakage for BF
POI_lst= [180, 181, 182, 183, 184, 185, 186, 188, 193, 194, 195, 197, 201, 203, 204, 205, 206, 207, 208, 210, 212, 214, 224, 228, 230, 232, 237, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 263, 264, 265, 266, 267, 268, 269, 270, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 304, 306, 307, 308, 309, 310, 311, 312, 313, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 349, 350, 351, 352, 353, 367, 368, 369, 370, 371]

# print(len(POI_lst), POI_lst)

# prepare label for ldr b
train_data = np.zeros([len(traindata), len(POI_lst)])
valid_data = np.zeros([len(validdata), len(POI_lst)])
train_labels = np.zeros([len(traindata), 1])
valid_labels = np.zeros([len(validdata), 1])
for i in range(len(traindata)):
    for j in range(len(POI_lst)):
        train_data[i][j] = traindata[i][POI_lst[j]]
    train_labels[i] = int(trainlabels[i][3]) + 1

for i in range(len(validdata)):
    for j in range(len(POI_lst)):
        valid_data[i][j] = validdata[i][POI_lst[j]]
    valid_labels[i] = int(validlabels[i][3]) + 1


test_data = np.zeros([len(testdata), len(POI_lst)])
test_labels = np.zeros([len(testdata), 1])
for i in range(len(testdata)):
    for j in range(len(POI_lst)):
        test_data[i][j] = testdata[i][POI_lst[j]]
    test_labels[i] = int(testlabels[i][3]) + 1

model = Net(len(POI_lst), 3)
model.load_state_dict(torch.load('Net10.pth'))
new_curves_tensor = torch.tensor(test_data, dtype=torch.float32)
model.eval()

with torch.no_grad():
    outputs = model(new_curves_tensor)
    _, predicted_labels = torch.max(outputs, dim=1)

sr = 0
total = 0
err_lst = []
for i in range(len(test_labels)):
    total += 1
    if int(predicted_labels[i].item()) != int(test_labels[i].item()):
        print(test_labels[i].item(), int(predicted_labels[i].item()))
        err_lst.append(testlabels[i][0])
    else:
        sr += 1
print(f"[Net10 for ldr b] accuracy of model on test data = {sr/total*100}%")
err_lst = list(set(err_lst))
print("error list: ", len(err_lst), err_lst)
