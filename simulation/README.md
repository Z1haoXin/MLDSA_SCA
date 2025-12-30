## ELMO Simulation Experiments

### 1. Repository Structure
```
├── data/                        # datasets used in experiments
├── KE/                          # MVKE
|   ├── reslog/                  # MVKE run log files 
|   ├── CuSearch.cu              # MVKE (test Neq from 1 to a predefined number)
|   ├── MVKE.cu                  # CUDA-accelerated MVKE (test time)
|   └── plot.py                  # plot recovery trace
└── TA/   # template attack
    ├── match_res/               # Template matching results
    ├── mldsaSimulation/         # ELMO simulation
    ├── Trace/                   # generated traces
    ├── generate.py              # generate simulated traces from ELMO
    ├── template_attack_s.py     # Template attack targeting s1
    ├── template_attack_x.py     # Template attack targeting x
    ├── template_build_s.py      # Template construction for s1
    ├── template_build_x.py      # Template construction for x
    └── utils_ta.py              # Common utilities for template attacks
```

### 2. Run simulation experiments
#### 2.1 Template attack under different noise level
To capturing simulated traces from ELMO, run:
```
cd ./TA
make -C mldsaSimulation
python3 generate.py
```
The generated simulation traces will stored in ./Trace.

Then, run
```
python3 template_build_s.py
python3 template_build_x.py
```
to build HW template for $\bm{s_1}$ and $\bm{x}$.

and run
```
python3 template_attack_s.py
python3 template_attack_x.py
```
to match the HW results for $\bm{s_1}$ and $\bm{x}$, the results will be stored in ./match_res.

#### 2.2 MVKE under different noise level
To obtain the recovery tendency, run
```
nvcc -O3 CuSearch.cu -lhdf5 -o CuSearch
./CuSearch 0.2 0 200
./CuSearch 0.2 1 200
./CuSearch 0.4 0 500
```
for ``./CuSearch 0.2 0 200`` the first parameter is the noise level ($\sigma$), the second parameter is the serial number of match_res file, and the third parameter is the total number of used equations (equals to $N_{eq}$ in the paper).
This program will execute MVKE with different $N_{eq}$ ( from 1 to $N_{eq}$) and count the correctly recovered coefficients. 

To test the run time, run
```
nvcc -O3 MVKE.cu -lhdf5 -o MVKE
./MVKE 0.2 0 200
```
for ``./MVKE 0.2 0 200`` the first parameter is the noise level ($\sigma$), the second parameter is the serial number of match_res file, and the third parameter is the total number of used equations (equals to $N_{eq}$ in the paper). This program will execute MVKE with $N_{eq}$, counting the correctly recovered coefficients and providing the time elapsed. 