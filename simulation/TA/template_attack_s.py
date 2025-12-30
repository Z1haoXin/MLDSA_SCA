from tqdm import trange,tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils_ta import *
import h5py 

def add_noise(traces, sigma=1e-4):
    noise = np.random.normal(0, sigma, traces.shape).astype(traces.dtype)
    return traces + noise

if __name__ == "__main__":

    trace_path = "Trace/"
    
    nb_cls = 33
    nb_cls_trc = 500
    rp = 6
    GE = 5
    sigma = 0.0001

    profile_trace = np.zeros((nb_cls*nb_cls_trc, 250))
    for i in range(rp):
        with h5py.File(trace_path + f"profile_trace_s1_{i}.h5", "r") as f:
            trace = f["traces"][:, :250]
        profile_trace += add_noise(trace, sigma)
    profile_trace /= rp

    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = [12, 26]
    # rep 1 [12, 26] 99.94
    # rep 2 [12, 26] 99.94
    # rep 4 [12, 26] 99.94
    # rep 6 [12, 26] 99.94
    template_s1hat = template_build(profile_trace, cls, POI_lst)
    for q in range(4):
        sr = []
        hitcnt = 0
        h5_file = h5py.File(f"match_res/match_res_shat_rk{GE}_no{q}.h5", "w")
        fhw_res = np.zeros((256, GE))
        for idx in range(256):
            nb_attack = 1000
            
            with h5py.File(trace_path + f"attack_trace_cs{q}.h5", "r") as f:
                trace = f[f"{idx}/traces"][:, :250]
            trace = add_noise(trace, sigma)

            with h5py.File("../data/attack_data_cs1_simu.h5", "r") as f:
                attack_ans = f["hws"][q, idx]
                attack_ans = attack_ans.tolist()

            nb_attack = 1000 // rp
            attack_trace = np.zeros((nb_attack, 250))
            for i in range(nb_attack):
                for j in range(rp):
                    attack_trace[i] += trace[i * rp + j]
                attack_trace[i] /= rp
                        
            match_res = template_matching(attack_trace, template_s1hat, POI_lst)
            hw_res = np.zeros((nb_attack, GE))
            prob_res = np.zeros((nb_attack, GE))
            succ = 0
            for No_trace in range(nb_attack):
                p = match_res[No_trace]
                
                # prob_res[No_trace] = np.array([ip/sum(p) for ip in p])
                # hw_res[No_trace] = np.array([i for i in range(33)])
                p_sorted = sorted(p)
                value = p_sorted[-GE:]
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
                for i in range(GE):
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
        h5_file.create_dataset("hw", data=fhw_res)
        h5_file.close()
            # print(f"idx={idx} [attack] success rate GE={GE}:{succ_rate}") 
        print(f"ave sr ge={GE}: {np.average(sr)}")
        # print(sr)
