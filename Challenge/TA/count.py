import h5py
import numpy as np

with h5py.File("trace/attack_data_challenge.h5", "r") as f:
    ans = f["hw"][:]

sr_lst = []
N = 256
for q in range(1000):
    sr = 0
    for idx in range(N):
        with h5py.File(f"res/match_res_c_idx_{idx}.h5", "r") as f:
            res = f["data"][q]
        if ans[q][idx] in res:
            sr += 1
    # print(f"No.{q} sr={sr*1.0/N}")
    sr_lst.append(sr*1.0/N) 

print("===summarize===")
print(f"min: {np.argmin(sr_lst)} {sr_lst[np.argmin(sr_lst)]}")
print(f"ave:      {np.average(sr_lst)}")
print(f"max: {np.argmax(sr_lst)} {sr_lst[np.argmax(sr_lst)]}")

   
    
