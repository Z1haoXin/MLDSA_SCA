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

    nb_test = 5000
    with h5py.File(trace_path + f"test_trace_x.h5", "r") as f:
        test_trace = f["trace"][:]

    with h5py.File(trace_path + "test_data_xhat.h5", "r") as f:
        test_ans = f["hw"][:]
        test_ans = test_ans.tolist()

    assert len(profile_trace) == nb_cls * nb_cls_trc
    assert len(test_trace) == nb_test
    assert len(test_ans) == nb_test

    # CPA to find the most correlation location index, as poi (point of interest)
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = select_POI(profile_trace, cls, "CPA", 0.1, True) # 0.1
    print(POI_lst.tolist())

    # POI_lst_selected = POI_lst.tolist()
    # tmp_lst = []
    # while tmp_lst != POI_lst_selected:
    #     tmp_lst = POI_lst_selected
    #     POI_lst_selected = POI_update(profile_trace, cls, POI_lst_selected, test_trace, test_ans, RK)

    # POI_lst = POI_lst_selected
    POI_lst = [251, 252, 261, 262, 263, 264]

    template_xhat = template_build(profile_trace, cls, POI_lst)
    match_res = template_matching(test_trace, template_xhat, POI_lst)

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
