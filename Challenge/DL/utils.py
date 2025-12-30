import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, x, y):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x, 32)
        self.fc2 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, y)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc2(self.fc1(x))))
        x = F.relu(self.bn2(self.fc3(x)))
        x = self.fc4(x)
        return F.softmax(x, dim=1)
    
class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.flatten(), dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def mean(X):
    return np.sum(X, axis=0)/len(X)

def std_dev(X, X_bar):
    return np.sqrt(np.sum((X-X_bar)**2, axis=0))

def cov(X, X_bar, Y, Y_bar):
    return np.sum((X-X_bar)*(Y-Y_bar), axis=0)

# POI selection via CPA
def POI_selection_via_CPA(traces, hws, draw=False, cor=0.5):
    t_bar = np.sum(traces, axis=0)/len(traces)
    o_t = np.sqrt(np.sum((traces - t_bar)**2, axis=0))
    hws_bar = mean(hws)
    o_hws = std_dev(hws, hws_bar)
    correlation = cov(traces, t_bar, hws, hws_bar)
    cpaoutput = correlation/(o_t*o_hws)
    if draw:
        plt.figure()
        plt.plot(abs(cpaoutput))
        plt.show()
    
    t = abs(cpaoutput)
    idxa = np.where(t>=cor)
    return idxa[0]