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
    sigma = 0.2 # 0.2 - 1.3 
    
    # POI_lst = [193, 194, 200, 201]

    with h5py.File(trace_path + f"profile_trace_x.h5", "r") as f:
        profile_trace = f["traces"][:, :250]

    nb_test = 5000
    with h5py.File(trace_path + f"test_trace_x.h5", "r") as f:
        test_trace = f["traces"][:, :250]
    
    profile_trace = normalize_traces(profile_trace)
    profile_trace = add_noise(profile_trace, sigma)
    
    
    test_trace = normalize_traces(test_trace)
    tmp = test_trace.copy()
    test_trace = add_noise(test_trace, sigma)

    with h5py.File("../data/test_data_xhat.h5", "r") as f:
        test_ans = f["hw"][:]
        test_ans = test_ans.tolist()

    assert len(profile_trace) == nb_cls * nb_cls_trc
    assert len(test_trace) == nb_test
    assert len(test_ans) == nb_test

    # CPA to find the most correlation location index, as poi (point of interest)
    cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
    POI_lst = select_POI(profile_trace, cls, "CPA", 0.05, True) # 0.1
    print(POI_lst.tolist())

    POI_lst_selected = POI_lst.tolist()
    tmp_lst = []
    while tmp_lst != POI_lst_selected:
        tmp_lst = POI_lst_selected
        POI_lst_selected = POI_update(profile_trace, cls, POI_lst_selected, test_trace, test_ans, GE)

    POI_lst =  POI_lst_selected
    print(compute_snr(tmp, test_trace, POI_lst))
    
    import time
    start_time = time.time()
    template_xhat = template_build(profile_trace, cls, POI_lst)
    end_time = time.time()

    execution_time = end_time - start_time
    print("elapse:", execution_time, "s")
    
    match_res = template_matching(test_trace, template_xhat, POI_lst)

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
