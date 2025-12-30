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

    for idx in range(256):
        with h5py.File(trc_path + f"profile_trace_challenge.h5", "r") as f:
            profile_trace = f[f"{idx}/trace"][:]
        
        nb_test = 5000
        with h5py.File(trc_path + f"test_trace_challenge.h5", "r") as f:
            test_trace = f[f"{idx}/trace"][:]

        with h5py.File(trc_path + "test_data_challenge.h5", "r") as f:
            test_ans = f["hw"][:, idx]
            test_ans = test_ans.tolist()

        # print(profile_trace.shape)

        # CPA to find the most correlation location index, as poi (point of interest)
        cls = np.array([i for i in range(nb_cls) for j in range(nb_cls_trc)])
        # POI_lst = select_POI(profile_trace, cls, "CPA", 0.05, False) # 0.75
        # print("POI: ", POI_lst)

        # POI_lst_selected = POI_lst.tolist()
        # tmp_lst = []
        # while tmp_lst != POI_lst_selected:
        #     tmp_lst = POI_lst_selected
        #     POI_lst_selected = POI_update(profile_trace, cls, POI_lst_selected, test_trace, test_ans, RK)
        # print(POI_lst_selected)
        # POI_lst = POI_lst_selected

        # with h5py.File(f"poi/poi_idx_{idx}.h5", "w") as f:
        #     f.create_dataset("poi", data=POI_lst)

        with h5py.File(trc_path + f"poi_challenge.h5", "r") as f:
            POI_lst = f[f"{idx}/poi"][:]

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
        print(f"idx={idx} [test] success rate RK={RK}:{succ_rate}") 
