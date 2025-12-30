from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

def normalize_traces(traces):
    traces = traces.copy()
    mean = np.mean(traces, axis=1, keepdims=True)
    std  = np.std(traces, axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return (traces - mean) / std

def add_noise(traces, sigma=1e-4):
    noise = np.random.normal(0, sigma, traces.shape).astype(traces.dtype)
    return traces + noise

def _calculate_cpa(traces, cls):
    """Calculate CPA metric (absolute correlation coefficient) for each sample point."""
    assert len(traces) == len(cls)

    trc_len = len(traces[0])
    cv = np.zeros(trc_len)
    for i in range(trc_len):
        C = np.corrcoef(traces[:, i], cls)
        cv[i] = abs(C[0, 1])  # Store absolute correlation values
    return cv


_METHOD_CONFIG = {
    'CPA': {
        'metric_func': _calculate_cpa,
    },
}

def select_POI(traces, cls, method='CPA', thd=0.5, draw=False):
    """
    Select Points of Interest (POI) using specified method
    
    Parameters:
    traces : ndarray
        Trace data matrix (nb_traces × nb_samples)
    method : str (default: 'CPA')
        Selection method (CPA)
    thd : float (optional)
        Threshold value for POI selection. Default 0.5
    draw : bool (default: False)
        Display metric plot
    cls : 
        - CPA: hws (hypothetical power values)

    Returns:
    ndarray : Array of POI indices
    """
    # Validate method
    if method not in _METHOD_CONFIG:
        raise ValueError(f"Unsupported method: {method}.")

    # Get method configuration
    config = _METHOD_CONFIG[method]
    
    # Validate required parameters
    if len(cls) == 0:
        raise ValueError(f"empty cls")

    # Calculate metric
    metric = config['metric_func'](traces, cls)
    
    # Apply comparison operator
    mask = metric >= thd
    poi_indices = np.where(mask)[0]

    # Visualize if requested
    if draw:
        plt.figure(figsize=(10, 4))
        plt.plot(metric)
        plt.title(f"{method} POI")
        plt.xlabel("Samples")
        plt.ylabel(f"{method} value")
        plt.axhline(y=thd, color='r', linestyle='--', label=f'Threshold ({thd})')
        plt.legend()
        plt.show()

    return poi_indices

def template_build(traces: np.ndarray, cls: np.ndarray, poi: list):
    """
    template building 
    
    params:
    traces : trace array (num_traces × num_samples)
    cls    : class label of each trace (num_traces,)
    poi    : points of interest
    
    return:
    tuple : (trace_ave, matrix_CM)
      - trace_ave : mean trace per class (num_classes × num_samples)
      - matrix_CM : cov matrix (num_poi × num_poi × num_classes)
    """
    assert len(traces) == len(cls), "trace number is not equal to class number"
    assert len(poi) > 0, "POI lst is empty"
    
    unique_cls, _, cls_counts = np.unique(cls, return_inverse=True, return_counts=True)
    num_cls = len(unique_cls)

    trace_ave = np.array([traces[cls == c].mean(axis=0) for c in unique_cls])
    
    noise = []
    for c_idx, c in enumerate(unique_cls):
        class_traces = traces[cls == c]
        class_noise = class_traces[:, poi] - trace_ave[c_idx, poi]
        noise.append(class_noise)

    num_poi = len(poi)
    matrix_CM = np.zeros((num_poi, num_poi, num_cls))
    for c_idx in range(num_cls):
        if cls_counts[c_idx] < 2:
            raise ValueError(f"class {unique_cls[c_idx]} no enough traces")
        matrix_CM[:, :, c_idx] = np.cov(noise[c_idx], rowvar=False)
    
    return trace_ave, matrix_CM

def template_matching(test_traces: np.ndarray, template: tuple, poi: list):
    """
    params:
    test_traces : test trace array (num_traces × num_samples)
    template    : template tuple (trace_ave, matrix_CM)
    poi         : POI lst

    return:
    ndarray : recorded matching result (num_traces × num_classes)
    """
    trace_ave, matrix_CM = template
    nb_test = test_traces.shape[0]
    num_cls = trace_ave.shape[0]
    num_poi = len(poi)
    
    inv_covs = np.empty_like(matrix_CM)
    log_dets = np.empty(num_cls)
    for c in range(num_cls):
        cov = matrix_CM[:, :, c]
        L = np.linalg.cholesky(cov) 
        inv_covs[:, :, c] = np.linalg.inv(L.T) @ np.linalg.inv(L)
        log_dets[c] = 2 * np.sum(np.log(np.diag(L)))

    matching_record = np.zeros((nb_test, num_cls))
    test_samples = test_traces[:, poi]
    
    for c in range(num_cls):
        diffs = test_samples - trace_ave[c, poi]
        quad_terms = np.einsum('ni,ij,nj->n', diffs, inv_covs[:, :, c], diffs)
        matching_record[:, c] = -0.5 * (num_poi * np.log(2 * np.pi) + log_dets[c] + quad_terms)

    return matching_record

def POI_update(profiled_traces, cls, POI_lst, test_traces, test_ans, RK = 5):
    tmp_lst = [0] + [i for i in POI_lst]
    selected_POI = [0] + [i for i in POI_lst]
    F_sr = 0
    nb_test = len(test_traces)
    for poi in tmp_lst:
        selected_POI.sort()
        selected_POI.remove(poi)
        template = template_build(profiled_traces, cls, selected_POI)
        match_res = template_matching(test_traces,template, selected_POI)
        succ = 0
        for No_trace in range(nb_test):
            p = match_res[No_trace]
            p_sorted = sorted(p)
            value = p_sorted[-RK:]
            index = [int(np.where(j == p)[0]) for j in value]
            if test_ans[No_trace] in index:
                succ+=1
        succ_rate = succ / nb_test
        print(f"success rate RK={RK}:{succ_rate}") 
        if F_sr <= succ_rate:
            F_sr = succ_rate
            print(f"Update SR:{F_sr} POI_lst:{selected_POI} remove:{poi}")
        else:
            selected_POI.append(poi)
    return selected_POI


def POI_update(profiled_traces, cls, POI_lst, test_traces, test_ans, GE = 5):
    tmp_lst = [0] + [i for i in POI_lst]
    selected_POI = [0] + [i for i in POI_lst]
    F_sr = 0
    nb_test = len(test_traces)
    for poi in tmp_lst:
        selected_POI.sort()
        selected_POI.remove(poi)
        template = template_build(profiled_traces, cls, selected_POI)
        match_res = template_matching(test_traces,template, selected_POI)
        succ = 0
        for No_trace in range(nb_test):
            p = match_res[No_trace]
            p_sorted = sorted(p)
            value = p_sorted[-GE:]
            index = [int(np.where(j == p)[0]) for j in value]
            if test_ans[No_trace] in index:
                succ+=1
        succ_rate = succ / nb_test
        print(f"success rate GE={GE}:{succ_rate}") 
        if F_sr <= succ_rate:
            F_sr = succ_rate
            print(f"Update SR:{F_sr} POI_lst:{selected_POI} remove:{poi}")
        else:
            selected_POI.append(poi)
    return selected_POI

