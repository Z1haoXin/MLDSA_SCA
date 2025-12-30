## DL-based profiling attack agianst challenges $c$

### 1. Repository Structure
```
├── cwcapture/
│   ├── firmware/ # ML-DSA firmware flashed onto the target device via ChipWhisperer
│   └── Challenge_SCA.ipynb       # ChipWhisperer trace acquisition notebook
├── DL/           # DL-based Profiling attack against first three rounds of NTT(c) 
|   ├── Fs/  # Datasets & codes for training & testing networks - first round 
|   ├── Sec/ # Datasets & codes for training & testing networks - second round 
|   ├── Trd/ # Datasets & codes for training & testing networks - third round 
|   └── utils.py # Common utilities for networks
├── search/ 
|   ├── bn_search.cpp   # Searching method recovering challenges
|   ├── count.py        # Statistics of experimental results
├── TA/                 # Template attack targeting chat
|   ├── res/            # Template matching results
|   ├── trace/          # Traces and datasets
|   ├── count.py        # Statistics of experimental results
|   ├── template_attack_c.py
|   ├── template_build_c.py
|   └── utils_ta.py

```
### 2. run the code

#### 2.1 DL-based profiling attack against the first three rounds of NTT(c)
Capturing traces from ChipWhisperer, and then store the ``.h5`` files of traces and datasets in ./DL/Fs, ./DL/Sec and ./DL/Trd.

For network training, `cd ./DL` and run:

```
cd Fs
python3 train_Net10.py
python3 train_Net11.py
```
and run:
```
python3 test_Net10.py
python3 test_Net11.py
```
to test the accuracy of networks.

For the second and third round of NTT, the networks are trained by running:
```
cd Sec
python3 train_Net2.py
python3 test_Net2.py
```
and
```
cd Trd
python3 train_Net3.py
python3 test_Net3.py
```


#### 2.2 Template attack against $\hat{c}$
`cd ./TA` and run
```
python3 template_build_c.py
python3 template_attack_c.py
```
to build and match the HW results for each $\hat{c}[i]$, the results will be stored in ./res.

#### 2.3 Recovering challenge via searching method
`cd ./search` and run
```
sudo apt update
sudo apt install libhdf5-serial-dev
g++ -o3 bn_search.cpp -o bn_search -lhdf5_serial
./bn_search > search_out.log
```