import h5py, sys, getopt
from random import gauss, choices
import numpy as np
import string
from random import randint
import matplotlib.pyplot as plt
from tqdm import trange

def generate_profile_traces(nb_cls, nb_traces, cdata, s1data, input_file):

    foldername = 'mldsaSimulation'
    classname = 'mldsa'

    from elmo.manage import get_simulation
    simu = get_simulation(classname, foldername)
    simulation = simu()

    trace_array = np.zeros((nb_cls * nb_traces, 420), dtype=np.float64)
    for i in trange(nb_cls):
        challenges = []
        for j in range(nb_traces):
            challenges.append([int(cdata[i][j]), int(s1data[i][j])])
        simulation.set_challenges(challenges)
        simulation.run()
        traces = simulation.get_traces()
        traces_np = np.array([np.pad(t, (0, 420-len(t)), 'constant') for t in traces], dtype=np.float64)

        trace_array[i*nb_traces:(i+1)*nb_traces] = traces_np

    with h5py.File(input_file, "w") as f:
        f.create_dataset("traces", data=trace_array)

def generate_test_traces(nb_traces, cdata, s1data, input_file):

    foldername = 'mldsaSimulation'
    classname = 'mldsa'

    from elmo.manage import get_simulation
    simu = get_simulation(classname, foldername)
    simulation = simu()

    trace_array = np.zeros((nb_traces, 420), dtype=np.float64)
    for i in trange(nb_traces//500):
        challenges = []
        for j in range(500):
            challenges.append([int(cdata[i*500+j]), int(s1data[i*500+j])])
        simulation.set_challenges(challenges)
        simulation.run()
        traces = simulation.get_traces()
    
        traces_np = np.array([np.pad(t, (0, 420-len(t)), 'constant') for t in traces], dtype=np.float64)

        trace_array[i*500:(i+1)*500] = traces_np

    with h5py.File(input_file, "w") as f:
        f.create_dataset("traces", data=trace_array)

def generate_attack_traces(q, nb_trace, cdata, s1data, input_file):

    foldername = 'mldsaSimulation'
    classname = 'mldsa'

    from elmo.manage import get_simulation
    simu = get_simulation(classname, foldername)
    simulation = simu()

    f = h5py.File(input_file, "w")
    for idx in trange(256):
        trace_array = np.zeros((nb_trace, 420), dtype=np.float64)
        g = f.create_group(f"{idx}")
        for i in range(nb_trace//500):
            challenges = []
            for j in range(500):
                challenges.append([int(cdata[i*500+j][idx]), int(s1data[q][idx])])
            simulation.set_challenges(challenges)
            simulation.run()
            traces = simulation.get_traces()
            traces_np = np.array([np.pad(t, (0, 420-len(t)), 'constant') for t in traces], dtype=np.float64)
            trace_array[i*500:(i+1)*500] = traces_np
        g.create_dataset("traces", data=trace_array)
    f.close()

def load_dataset(filename, place):
    with h5py.File(filename, "r") as f:
        data = f[place][:]
    return data


def main():

    # profile s
    print("generate profile traces of s")
    s1data = load_dataset("../data/profile_data_s1hat.h5", "/data")
    for r in range(6):
        cdata =  np.random.randint(-1-4*8380417, 2+4*8380417, size=s1data.shape, dtype=np.int64)
        generate_profile_traces(33, 500, cdata, s1data, f"Trace/profile_trace_s1_{r}.h5")
    
    # test s
    print("generate test traces of s")
    s1data = load_dataset("../data/test_data_s1hat.h5", "/data")
    for r in range(6):
        cdata =  np.random.randint(-1-4*8380417, 2+4*8380417, size=s1data.shape, dtype=np.int64)
        generate_test_traces(5000, cdata, s1data, f"Trace/test_trace_s1_{r}.h5")

    # # profile x
    print("generate profile traces of x")
    cdata = load_dataset("../data/profile_data_xhat.h5", "/c_data")
    s1data = load_dataset("../data/profile_data_xhat.h5", "/s1_data")
    generate_profile_traces(33, 500, cdata, s1data,  "Trace/profile_trace_x.h5")

    # # test x
    print("generate test traces of x")
    cdata = load_dataset("../data/test_data_xhat.h5", "/c_data")
    s1data = load_dataset("data/test_data_xhat.h5", "/s1_data")
    generate_test_traces(5000, cdata, s1data,  "Trace/test_trace_x.h5")

    # # attack s / x
    
    cdata = load_dataset("../data/attack_data_cs1_simu.h5", "/chat")
    s1data = load_dataset("../data/attack_data_cs1_simu.h5", "/s1hat")
    print("generate attack traces of cs0")
    generate_attack_traces(0, 3000, cdata, s1data,  "Trace/attack_trace_cs0.h5")
    print("generate attack traces of cs1")
    generate_attack_traces(1, 3000, cdata, s1data,  "Trace/attack_trace_cs1.h5")
    print("generate attack traces of cs2")
    generate_attack_traces(2, 3000, cdata, s1data,  "Trace/attack_trace_cs2.h5")
    print("generate attack traces of cs3")
    generate_attack_traces(3, 3000, cdata, s1data,  "Trace/attack_trace_cs3.h5")
    

if __name__ == "__main__":
    main()
    
