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
    rp = 1
    GE = 5
    sigma = 0.1

    profile_trace = np.zeros((nb_cls*nb_cls_trc, 250))
    for i in range(rp):
        with h5py.File(trace_path + f"profile_trace_s1_{i}.h5", "r") as f:
            trace = f["traces"][:, :250]
        trace = normalize_traces(trace)
        profile_trace += add_noise(trace, sigma)
    profile_trace /= rp
        
        
    nb_test = 5000
    test_trace = np.zeros((nb_test, 250))
    for i in range(rp):
        with h5py.File(trace_path + f"test_trace_s1_{i}.h5", "r") as f:
            trace = f["traces"][:, :250]
        trace = normalize_traces(trace)
        test_trace += add_noise(trace, sigma)
    test_trace /= rp

    with h5py.File("../data/test_data_s1hat.h5", "r") as f:
        test_ans = f["hw"][:]
        test_ans = test_ans.tolist()

    assert len(profile_trace) == nb_cls * nb_cls_trc
    assert len(test_trace) == nb_test
    assert len(test_ans) == nb_test
    
    # CPA to find the most correlation location index, as poi (point of interest)
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = select_POI(profile_trace, cls, "CPA", 0.4, True) # 0.75
    print("POI: ", POI_lst)
    POI_lst_selected = POI_lst.tolist()
    tmp_lst = []
    while tmp_lst != POI_lst_selected:
        tmp_lst = POI_lst_selected
        POI_lst_selected = POI_update(profile_trace, cls, POI_lst_selected, test_trace, test_ans, GE)

    POI_lst = POI_lst_selected
    # rep 1 [12, 26] 99.94
    # rep 2 [12, 26] 99.94
    # rep 4 [12, 26] 99.94
    # rep 6 [12, 26] 99.94
    template_s1hat = template_build(profile_trace, cls, POI_lst)

    match_res = template_matching(test_trace, template_s1hat, POI_lst)
    hw_res = np.zeros((nb_test, GE))
    prob_res = np.zeros((nb_test, GE))
    succ = 0
    for No_trace in range(nb_test):
        p = match_res[No_trace]
        p_sorted = sorted(p)
        value = p_sorted[-GE:]
        index = [int(np.where(j == p)[0]) for j in value]
        prob_res[No_trace] = np.array(value)
        hw_res[No_trace] = np.array(index)
        if test_ans[No_trace] in index:
            succ += 1
        
    succ_rate = succ / nb_test
    print(f"success rate GE={GE}:{succ_rate}") 
