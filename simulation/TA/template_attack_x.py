from tqdm import trange,tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils_ta import *
import h5py

if __name__ == "__main__":

    trace_path = "Trace/"

    nb_cls = 33
    nb_cls_trc = 500
    GE = 5
    sigma = 0.4
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = [117, 118, 132, 136, 142, 143, 159, 160, 161, 171, 172, 174, 175, 176, 177, 178, 179, 189, 193, 194, 198, 199, 200, 201]
    # sigma=0.2 [117, 131, 142, 143, 171, 172, 174, 175, 176, 177, 178, 179, 198, 200, 201]
    # sigma=0.4 [117, 118, 132, 136, 142, 143, 159, 160, 161, 171, 172, 174, 175, 176, 177, 178, 179, 189, 193, 194, 198, 199, 200, 201]
    # sigma = 0.6 [117, 118, 135, 142, 143, 161, 171, 172, 174, 175, 176, 177, 178, 179, 200, 201]
    # sigma=0.8 [25, 118, 131, 132, 135, 142, 143, 158, 159, 161, 171, 172, 174, 175, 176, 177, 178, 179, 193, 200, 201] 

    with h5py.File(trace_path + f"profile_trace_x.h5", "r") as f:
        profile_trace = f["traces"][:, :250]
    profile_trace = normalize_traces(profile_trace)
    profile_trace = add_noise(profile_trace, sigma)

    template_xhat = template_build(profile_trace, cls, POI_lst)
    total_sr = 0
    nb_no = 5
    for no in range(nb_no):
        for q in range(4):
            sr = []
            h5_file = h5py.File(f"match_res/match_res_xhat_rk{GE}_no{q}_sigma{sigma}_{no}.h5", "w")
            for idx in range(256):
                g = h5_file.create_group(f"{idx}")
                nb_attack = 3000
                with h5py.File(trace_path + f"attack_trace_cs{q}.h5", "r") as f:
                    attack_trace = f[f"{idx}/traces"][:, :250]
                attack_trace = normalize_traces(attack_trace)
                attack_trace = add_noise(attack_trace, sigma)

                with h5py.File("../data/attack_data_cs1_simu.h5", "r") as f:
                    attack_ans = f["hwx"][q, :, idx]
                    attack_ans = attack_ans.tolist()
                
                match_res = template_matching(attack_trace, template_xhat, POI_lst)
                hw_res = np.zeros((nb_attack, GE))
                prob_res = np.zeros((nb_attack, GE))
                succ = 0
                for No_trace in range(nb_attack):
                    p = match_res[No_trace]
                    p_sorted = sorted(p)
                    value = p_sorted[-GE:]
                    index = [int(np.where(j == p)[0]) for j in value]
                    prob_res[No_trace] = np.array(value)
                    hw_res[No_trace] = np.array(index)
                    if attack_ans[No_trace] in index:
                        succ += 1
                succ_rate = succ / nb_attack
                sr.append(succ_rate)
                g.create_dataset("hw", data = hw_res)
                # print(f"idx = {idx} [attack] success rate GE={GE}:{succ_rate}") 
            h5_file.close()
            print(f"ave sr ge={GE}: {np.average(sr)}")
            total_sr += np.average(sr)
    print(f"total ave sr = {total_sr/(4*nb_no)}")
            