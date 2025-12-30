from tqdm import trange,tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils_ta import *
import hdf5plugin, h5py

if __name__ == "__main__":

    trace_path = "Trace/"

    nb_cls = 33
    nb_cls_trc = 500
    RK = 5

    with h5py.File(trace_path + f"profile_trace_x.h5", "r") as f:
        profile_trace = f["trace"][:]
    
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = [251, 252, 261, 262, 263, 264]

    template_xhat = template_build(profile_trace, cls, POI_lst)

    for q in range(4):
        sr = []
        # h5_file = h5py.File(f"match_res_xhat_rk{RK}_no{q}.h5", "w")
        for idx in range(256):
            # g = h5_file.create_group(f"{idx}")
            nb_attack = 1000
            with h5py.File(trace_path + f"attack_trace_cs1_{q}.h5", "r") as f:
                attack_trace = f[f"{idx}/trace"][:]

            with h5py.File(trace_path + "attack_data_cs1.h5", "r") as f:
                attack_ans = f["hwx"][q, :, idx]
                attack_ans = attack_ans.tolist()
 
            match_res = template_matching(attack_trace, template_xhat, POI_lst)
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
            sr.append(succ_rate)
            # g.create_dataset("hw", data = hw_res)
            # g.create_dataset("prob", data=prob_res)
            # # print(f"idx = {idx} [attack] success rate RK={RK}:{succ_rate}") 
        # h5_file.close()
        print(f"ave sr ge={RK}: {np.average(sr)}")
        