## Template attack agianst $\{s_1}$ and $\{x}$

### 1. Repository Structure
```
├── cwcapture/
│   ├── firmware/ # ML-DSA firmware flashed onto the target device via ChipWhisperer
│   └── MLDSA_SCA.ipynb       # ChipWhisperer trace acquisition notebook
├── match_res/               # Template matching results
├── Trace/                   # Collected traces and datasets
├── template_attack_s.py     # Template attack targeting s1
├── template_attack_x.py     # Template attack targeting x
├── template_build_s.py      # Template construction for s1
├── template_build_x.py      # Template construction for x
└── utils_ta.py              # Common utilities for template attacks

```
### 2. run the code
Capturing traces from ChipWhisperer, and then store the ``.h5`` files of traces and datasets in ./Trace.
#### 2.1 building templates for $\bm{s_1}$ and $\bm{x}$
run
```
python3 template_build_s.py
python3 template_build_x.py
```
to build HW template for $\bm{s_1}$ and $\bm{x}$.

#### 2.2 Template matching for $\bm{s_1}$ and $\bm{x}$
run
```
python3 template_attack_s.py
python3 template_attack_x.py
```
to match the HW results for $\bm{s_1}$ and $\bm{x}$, the results will be stored in ./match_res.
