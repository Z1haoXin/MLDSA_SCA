## Majority Voting based Key Enumeration (MVKE)

### 1. Repository Structure
```
├── res/ # searching results of MVKE
├──  CuSearch.cu             # CUDA-accelerated MVKE (test Neq from 1 to a predefined number)
└──  MVKE.cu                 # CUDA-accelerated MVKE (test time)


```
### 2. run the code

#### 2.1 run the recovery trace (Fig.8 in the paper)
run
```
nvcc -O3 CuSearch.cu -lhdf5 -o CuSearch
./CuSearch 0 100
```
for ``./CuSearch 0 100`` the first parameter is the begin idx of constraint equation, and the second parameter is the total number of used equations (equals to $N_{eq}$ in the paper).
This program will execute MVKE with different $N_{eq}$ ( from 1 to $N_{eq}$) and count the correctly recovered coefficients. 

#### 2.2 testing the time spend

run
```
nvcc -O3 MVKE.cu -lhdf5 -o MVKE
./MVKE 0 100
```
for ``./MVKE 0 100`` the first parameter is the begin idx of constraint equation, and the second parameter is the total number of used equations (equals to $N_{eq}$ in the paper). This program will execute MVKE with $N_{eq}$, counting the correctly recovered coefficients and providing the time elapsed. 