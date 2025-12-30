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
    rp = 6
    RK = 5

    profile_trace = np.zeros((nb_cls*nb_cls_trc, 500))
    for i in range(rp):
        with h5py.File(trace_path + f"profile_trace_s_{i}.h5", "r") as f:
            trace = f["trace"][:]
        profile_trace += trace
    profile_trace /= rp

    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = [32, 33, 34, 40, 45, 46, 47, 49, 50, 53, 54, 55, 61, 69, 105, 106, 107, 109, 110, 111, 112, 113, 120, 121, 122, 125, 126, 127, 147, 156, 188, 189]
    # rep 1 [123, 124, 125, 126, 127] 96.30
    # rep 2 [121, 122, 123, 124, 125, 126] 97.44
    # rep 3 [32, 33, 34, 38, 45, 56, 57, 123, 124, 125, 126, 127, 147] 97.68
    # rep 4 [32, 33, 34, 35, 36, 37, 38, 56, 59, 60, 106, 107, 110, 122, 125, 126, 127, 187, 188] 98.92
    # rep 6 [32, 33, 34, 40, 45, 46, 47, 49, 50, 53, 54, 55, 61, 69, 105, 106, 107, 109, 110, 111, 112, 113, 120, 121, 122, 125, 126, 127, 147, 156, 188, 189] 99.82
    template_s1hat = template_build(profile_trace, cls, POI_lst)
    for q in range(4):
        sr = []
        hitcnt = 0
        # h5_file = h5py.File(f"match_res_shat_rk{RK}_no{q}.h5", "w")
        fhw_res = np.zeros((256, RK))
        for idx in range(256):
            nb_attack = 1000
            
            with h5py.File(trace_path + f"attack_trace_cs1_{q}.h5", "r") as f:
                trace = f[f"{idx}/trace"][:]

            with h5py.File(trace_path + "attack_data_cs1.h5", "r") as f:
                attack_ans = f["hws"][q, idx]
                attack_ans = attack_ans.tolist()

            nb_attack = 1000 // rp
            attack_trace = np.zeros((nb_attack, 500))
            for i in range(nb_attack):
                for j in range(rp):
                    attack_trace[i] += trace[i * rp + j]
                attack_trace[i] /= rp
                        
            match_res = template_matching(attack_trace, template_s1hat, POI_lst)
            hw_res = np.zeros((nb_attack, RK))
            prob_res = np.zeros((nb_attack, RK))
            succ = 0
            for No_trace in range(nb_attack):
                p = match_res[No_trace]
                
                # prob_res[No_trace] = np.array([ip/sum(p) for ip in p])
                # hw_res[No_trace] = np.array([i for i in range(33)])
                p_sorted = sorted(p)
                value = p_sorted[-RK:]
                index = [int(np.where(j == p)[0]) for j in value]
                prob_res[No_trace] = np.array(value)
                hw_res[No_trace] = np.array(index)
                if attack_ans in index:
                    succ += 1
                
            succ_rate = succ / nb_attack
            sr.append(succ_rate)

            fhw = []
            fprob = []
            for j in range(3):
                prob_res[j] = prob_res[j] / np.sum(prob_res[j])
                for i in range(RK):
                    if hw_res[j][i] not in fhw:
                        fhw.append(hw_res[j][i])
                        fprob.append(prob_res[j][i])
                    else:
                        ind = fhw.index(hw_res[j][i])
                        fprob[ind] += prob_res[j][i]
            fhw = sorted(fhw, key=lambda x: fprob[fhw.index(x)])
            fprob = sorted(fprob)
            # print(fhw)
            # print(fprob)
            fhw_res[idx] = np.array(fhw[-5:])
            if attack_ans in fhw[-5:]:
                hitcnt += 1

            # print(f"s{q}[{idx}]: hws = {attack_ans}, hwlist = {fhw[-5:]}")
        print(f"KEY{q}: SR = {hitcnt/256}")
        # h5_file.create_dataset("hw", data=fhw_res)
        # h5_file.close()
            # print(f"idx={idx} [attack] success rate RK={RK}:{succ_rate}") 
        print(f"ave sr ge={RK}: {np.average(sr)}")
        # print(sr)
