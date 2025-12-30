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
    testdata = f["test/data"][:]
    testlabel = f["test/label"][:]

print(len(realdata), len(realdata[0]))
cnt = [0,0,0]
cnt0 = [0,0,0]
for i in range(len(realdata)):
    if real_label[i][0] < 2250:
        cnt[real_label[i][2]] += 1
print(cnt)

traindata0 = np.zeros([10083, 1000])
trainlabel0 = np.zeros([10083, 8])
traindata1 = np.zeros([112568, 1000])
trainlabel1 = np.zeros([112568, 8])
traindata2 = np.zeros([10225, 1000])
trainlabel2 = np.zeros([10225, 8])

traindata = [traindata0, traindata1, traindata2]
trainlabel = [trainlabel0, trainlabel1, trainlabel2]

train_cnt = [0,0,0]
for i in range(len(realdata)):
    idx = real_label[i][2]
    if real_label[i][0] < 2250:
        traindata[idx][train_cnt[idx]] = realdata[i]
        trainlabel[idx][train_cnt[idx]] = real_label[i]
        train_cnt[idx] += 1

print(train_cnt)


err_lst = []
for NNno in [0,1,2]:

    tdata = traindata[NNno]
    tlabel = trainlabel[NNno]
    BOX = [
        [[14, 6],[21, 11]],
        [[0, 1],[0, 16],[0, 14]],
        [[11, 12],[12, 25]],
    ]
    tbox_lst = BOX[NNno]
    POI_lst = []
    for k in range(len(tbox_lst)):
        t = [tbox_lst[k][int(tlabel[i][3])] for i in range(len(tdata))] 
        POI = POI_selection_via_CPA(tdata,  np.array([t]).transpose(), False, 0.05)
        POI_lst += POI.tolist()
    POI_lst = sorted(list(set(POI_lst)))

    
    test_data = np.zeros([len(testdata), len(POI_lst)])
    test_labels = np.zeros([len(testdata), 1])
    for i in range(len(testdata)):
        for j in range(len(POI_lst)):
            test_data[i][j] = testdata[i][POI_lst[j]]
        test_labels[i] = int(testlabel[i][3])
    
   
    model = Net(len(POI_lst), 2)
    model.load_state_dict(torch.load(f'Net2{NNno}.pth'))
    

    new_curves_tensor = torch.tensor(test_data, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        outputs = model(new_curves_tensor)
        _, predicted_labels = torch.max(outputs, dim=1)
    
    sr = 0
    total = 0
    for i in range(len(test_labels)):
        if testlabel[i][2] == NNno:
            total+=1
            if int(predicted_labels[i].item()) != int(test_labels[i].item()):
                print(testlabel[i], int(predicted_labels[i].item()), int(test_labels[i].item()))
                err_lst.append(testlabel[i][0])
            else:
                sr+=1
    print(f"[ Net2{NNno} ] total = {total}, Accuracy of Net2{NNno} on test data = {sr/total*100}%")
err_lst = list(set(err_lst))
print("error list: ", len(err_lst), err_lst)