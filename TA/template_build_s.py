from tqdm import trange,tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils_ta import *
import hdf5plugin, h5py 
import argparse

if __name__ == "__main__":

    trace_path = "Trace/"
    
    nb_cls = 33
    nb_cls_trc = 500
    rp = 1 # range from 1 to 6
    RK = 5

    profile_trace = np.zeros((nb_cls*nb_cls_trc, 500))
    for i in range(rp):
        with h5py.File(trace_path + f"profile_trace_s_{i}.h5", "r") as f:
            trace = f["trace"][:]
        profile_trace += trace
    profile_trace /= rp
        
        
    nb_test = 5000
    test_trace = np.zeros((nb_test, 500))
    for i in range(rp):
        with h5py.File(trace_path + f"test_trace_s_{i}.h5", "r") as f:
            trace = f["trace"][:]
        test_trace += trace
    test_trace /= rp

    with h5py.File(trace_path + "test_data_s1hat.h5", "r") as f:
        test_ans = f["hw"][:]
        test_ans = test_ans.tolist()

    assert len(profile_trace) == nb_cls * nb_cls_trc
    assert len(test_trace) == nb_test
    assert len(test_ans) == nb_test
    
    # CPA to find the most correlation location index, as poi (point of interest)
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = select_POI(profile_trace, cls, "CPA", 0.4, True) # change 0.4 to other threshold
    print("POI: ", POI_lst)

    # greedy POI selection

    # POI_lst_selected = POI_lst.tolist()
    # tmp_lst = []
    # while tmp_lst != POI_lst_selected:
    #     tmp_lst = POI_lst_selected
    #     POI_lst_selected = POI_update(profile_trace, cls, POI_lst_selected, test_trace, test_ans, RK)

    # POI_lst = POI_lst_selected

    POI_lst = [123, 124, 125, 126, 127]
    # rep 1 [123, 124, 125, 126, 127] 96.30
    # rep 2 [121, 122, 123, 124, 125, 126] 97.44
    # rep 3 [32, 33, 34, 38, 45, 56, 57, 123, 124, 125, 126, 127, 147] 97.68
    # rep 4 [32, 33, 34, 35, 36, 37, 38, 56, 59, 60, 106, 107, 110, 122, 125, 126, 127, 187, 188] 98.92
    # rep 6 [32, 33, 34, 40, 45, 46, 47, 49, 50, 53, 54, 55, 61, 69, 105, 106, 107, 109, 110, 111, 112, 113, 120, 121, 122, 125, 126, 127, 147, 156, 188, 189] 99.82
    template_s1hat = template_build(profile_trace, cls, POI_lst)

    match_res = template_matching(test_trace, template_s1hat, POI_lst)
    hw_res = np.zeros((nb_test, RK))
    prob_res = np.zeros((nb_test, RK))
    succ = 0
    for No_trace in range(nb_test):
        p = match_res[No_trace]
        p_sorted = sorted(p)
        value = p_sorted[-RK:]
        index = [int(np.where(j == p)[0]) for j in value]
        prob_res[No_trace] = np.array(value)
        hw_res[No_trace] = np.array(index)
        if test_ans[No_trace] in index:
            succ += 1
        
    succ_rate = succ / nb_test
    print(f"success rate RK={RK}:{succ_rate}") 

