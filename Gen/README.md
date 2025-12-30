## Data Generation

### 1. Build the Extension Module

First, build the C/C++ extension to generate the shared object (`.so`) file, which will be imported by `Gen_TV.py`:

```
python3 setup.py build_ext --inplace
```

### 2. Generate Profiling Dataset
To generate the profiling dataset for the target $s_1$, run:
```
python3 Gen_TV.py --target="s1" --nb_cls=500 --datatype="uniform" --filename="profile_data_s1.h5"
```

### 3. Generate Test / Attack Dataset
To generate the test(attack) dataset for the target $s_1$, run:
```
python3 Gen_TV.py --target="s1" --nb_cls=1000 --datatype="random" --filename="test_data_s1.h5"
```

### 4. Other Targets
For other targets(e.g. $x$ and $c$), simply change the ``--target`` argument accordingly:
```
--target="x"
```
or
```
--target="c"
```