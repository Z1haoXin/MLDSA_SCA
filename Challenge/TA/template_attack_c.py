from tqdm import trange,tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils_ta import *
import hdf5plugin, h5py 



if __name__ == "__main__":

    trc_path = "trace/"
    nb_cls = 33
    nb_cls_trc = 500
    rp = 1
    RK = 5
    sr = []
    for idx in range(256):
        with h5py.File(trc_path + f"profile_trace_challenge.h5", "r") as f:
            profile_trace = f[f"{idx}/trace"][:]
            
        nb_attack = 1000
        with h5py.File(trc_path + f"attack_trace_challenge.h5", "r") as f:
            attack_trace = f[f"{idx}/trace"][:]

        with h5py.File(trc_path + "attack_data_challenge.h5", "r") as f:
            attack_ans = f["hw"][:, idx]
            attack_ans = attack_ans.tolist()

        # print(profile_trace.shape)

        cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
        with h5py.File(trc_path + f"poi_challenge.h5", "r") as f:
            POI_lst = f[f"{idx}/poi"][:]

        template_chat = template_build(profile_trace, cls, POI_lst)
        match_res = template_matching(attack_trace, template_chat, POI_lst)

        hw_res = np.zeros((nb_attack, RK))
        prob_res = np.zeros((nb_attack, RK))
        succ = 0
        for No_trace in range(nb_attack):
            p = match_res[No_trace]
            p_sorted = sorted(p)
            value = p_sorted[-RK:]
            index = [int(np.where(j == p)[0]) for j in value]
            prob_res[No_trace] = np.array(value)
            hw_res[No_trace] = np.array(index)
            if attack_ans[No_trace] in index:
                succ += 1
            
        succ_rate = succ / nb_attack
        print(f"idx={idx} [attack] success rate RK={RK}:{succ_rate}") 
        sr.append(succ_rate)
    print(f"ave sr = {np.mean(sr)}")
        # with h5py.File(f"res/match_res_c_idx_{idx}.h5", "w") as f:
        #     f.create_dataset("data", data=hw_res)