#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <hdf5/serial/hdf5.h>

#define Q 8380417
#define QINV 58728449
#define N 256
#define TEST_TRACE 1000
#define TOTAL_COEFF (4*N)

using namespace std;

int min_eqs, max_eqs;
int32_t chat[TEST_TRACE][N];
int32_t shat[4][N];
int32_t ldrs[4][N][5];
int32_t strx[4][N][TEST_TRACE][5];
double  prob[4][N][TEST_TRACE][5];


vector<int32_t> hw_candidate;
int hw_sizes[33];
int hw_offsets[33];

int32_t *d_chat;
int32_t *d_strx;
double  *d_prob;
int32_t *d_ldrs;
int32_t *d_candidate;
double  *d_hit;
int32_t *d_hw_candidate;
int *d_hw_sizes;
int *d_hw_offsets;

__host__ __device__ inline int32_t montgomery_reduce(int64_t a) {
    int32_t t = (int32_t)a * QINV;
    return (a - (int64_t)t * Q) >> 32;
}

bool load_data() {
    hid_t f = H5Fopen("../TA/Trace/attack_data_cs1.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return false;

    hid_t d = H5Dopen2(f, "/chat", H5P_DEFAULT);
    H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, chat);
    H5Dclose(d);

    d = H5Dopen2(f, "/s1hat", H5P_DEFAULT);
    H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shat);
    H5Dclose(d);

    H5Fclose(f);
    return true;
}

bool load_hw() {
    for (int q = 0; q < 4; q++) {
        char fn[128];
        sprintf(fn, "../TA/match_res/match_res_shat_rk5_no%d.h5", q);
        hid_t f = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (f < 0) return false;

        hid_t d = H5Dopen2(f, "/hw", H5P_DEFAULT);
        H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ldrs[q]);
        H5Dclose(d);
        H5Fclose(f);

        sprintf(fn, "../TA/match_res/match_res_xhat_rk5_no%d.h5", q);
        f = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (f < 0) return false;

        for (int i = 0; i < N; i++) {
            char path[32];
            sprintf(path, "/%d/hw", i);
            d = H5Dopen2(f, path, H5P_DEFAULT);
            H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, strx[q][i]);
            H5Dclose(d);

            sprintf(path, "/%d/prob", i);
            d = H5Dopen2(f, path, H5P_DEFAULT);
            H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prob[q][i]);
            H5Dclose(d);
        }
        H5Fclose(f);
    }
    return true;
}

void init_hw_candidate() {
    memset(hw_sizes, 0, sizeof(hw_sizes));
    for (int32_t a = -2 - 4*Q; a <= 2 + 4*Q; a++)
        hw_sizes[__builtin_popcount((uint32_t)a)]++;

    hw_offsets[0] = 0;
    for (int i = 1; i <= 32; i++)
        hw_offsets[i] = hw_offsets[i-1] + hw_sizes[i-1];

    hw_candidate.resize(hw_offsets[32] + hw_sizes[32]);
    int cnt[33] = {0};

    for (int32_t a = -2 - 4*Q; a <= 2 + 4*Q; a++) {
        int hw = __builtin_popcount((uint32_t)a);
        hw_candidate[hw_offsets[hw] + cnt[hw]++] = a;
    }
}

__global__ void mvke_prob_kernel(
    const int32_t* __restrict__ d_chat,
    const int32_t* __restrict__ d_strx,
    const double*  __restrict__ d_prob,
    const int32_t* __restrict__ d_ldrs,
    int trace_num,
    int min_eqs,
    const int32_t* __restrict__ d_hw_candidate,
    const int* __restrict__ d_hw_sizes,
    const int* __restrict__ d_hw_offsets,
    int32_t* d_candidate,
    double*  d_hit
){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int idx = bid & 255;
    int q   = bid >> 8;
    int global_idx = (q<<8) + idx;

    extern __shared__ unsigned char smem[];
    double*  s_hit  = (double*)smem;
    int32_t* s_cand = (int32_t*)(s_hit + blockDim.x);

    int base_ldrs = q*N*5 + idx*5;
    int hws[5];
    #pragma unroll
    for (int i=0;i<5;i++) hws[i] = d_ldrs[base_ldrs+i];

    double local_hit = 0.0;
    int32_t local_cand = 0;

    int base_strx = (q*N + idx) * TEST_TRACE * 5;
    int t_end = min(min_eqs + trace_num, TEST_TRACE);

    for (int k = 0; k < 5; k++) {
        int hw = hws[k];
        int start = d_hw_offsets[hw];
        int len   = d_hw_sizes[hw];

        for (int j = tid; j < len; j += blockDim.x) {
            int32_t a = d_hw_candidate[start + j];
            double vote = 0.0;

            for (int t = min_eqs; t < t_end; t++) {
                int32_t hwx = __popc((uint32_t)
                    montgomery_reduce((int64_t)d_chat[t*N + idx] * a));
                int off = base_strx + t*5;
                #pragma unroll
                for (int r=0;r<5;r++)
                    if (hwx == d_strx[off+r])
                        vote += d_prob[off+r];
            }

            if (vote > local_hit || (vote == local_hit && a < local_cand)) {
                local_hit = vote;
                local_cand = a;
            }
        }
    }

    s_hit[tid]  = local_hit;
    s_cand[tid] = local_cand;
    __syncthreads();

    if (tid == 0) {
        double best = -1.0;
        int32_t best_a = 0;
        for (int i=0;i<blockDim.x;i++) {
            if (s_hit[i] > best ||
               (s_hit[i] == best && s_cand[i] < best_a)) {
                best = s_hit[i];
                best_a = s_cand[i];
            }
        }
        d_hit[global_idx] = best;
        d_candidate[global_idx] = best_a;
    }
}

void majority_vote_gpu(int trace_num) {
    int threads = 256;
    int blocks  = TOTAL_COEFF;
    size_t shmem = threads * (sizeof(double) + sizeof(int32_t));

    cudaMemset(d_hit, 0, TOTAL_COEFF*sizeof(double));
    cudaMemset(d_candidate, 0, TOTAL_COEFF*sizeof(int32_t));

    mvke_prob_kernel<<<blocks, threads, shmem>>>(
        d_chat, d_strx, d_prob, d_ldrs,
        trace_num, min_eqs,
        d_hw_candidate, d_hw_sizes, d_hw_offsets,
        d_candidate, d_hit
    );
    cudaDeviceSynchronize();
}

void dfs_gpu_stats() {
    int32_t h_candidate[TOTAL_COEFF];
    double  h_hit[TOTAL_COEFF];

    for (int tr = 1; tr <= max_eqs; tr++) {
        majority_vote_gpu(tr);

        cudaMemcpy(h_candidate, d_candidate,
                   TOTAL_COEFF*sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hit, d_hit,
                   TOTAL_COEFF*sizeof(double), cudaMemcpyDeviceToHost);

        int recovered = 0;
        for (int i=0;i<TOTAL_COEFF;i++) {
            int q = i / 256;
            int n = i % 256;
            if (h_candidate[i] == shat[q][n] ||
                montgomery_reduce((int64_t)h_candidate[i]) ==
                montgomery_reduce((int64_t)shat[q][n]))
                recovered++;
        }

        printf("#trace_num=%d recovered=%d / %d\n",
               tr, recovered, TOTAL_COEFF);
        fflush(stdout);

        if (recovered == TOTAL_COEFF) break;
    }
}
void normalize_hw(){
    for(int i=0;i<4;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<TEST_TRACE;k++){
                double total = 0;
                for(int l=0;l<5;l++) total += prob[i][j][k][l];
                for(int l=0;l<5;l++) prob[i][j][k][l] /= total;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("usage: %s <min_eqs> <max_eqs>\n", argv[0]);
        return 0;
    }
    min_eqs = atoi(argv[1]);
    max_eqs = atoi(argv[2]);

    load_data();
    load_hw();
    normalize_hw();
    init_hw_candidate();

    cudaMalloc(&d_chat, TEST_TRACE*N*sizeof(int32_t));
    cudaMalloc(&d_strx, 4*N*TEST_TRACE*5*sizeof(int32_t));
    cudaMalloc(&d_prob, 4*N*TEST_TRACE*5*sizeof(double));
    cudaMalloc(&d_ldrs, 4*N*5*sizeof(int32_t));
    cudaMalloc(&d_candidate, TOTAL_COEFF*sizeof(int32_t));
    cudaMalloc(&d_hit, TOTAL_COEFF*sizeof(double));
    cudaMalloc(&d_hw_candidate, hw_candidate.size()*sizeof(int32_t));
    cudaMalloc(&d_hw_sizes, 33*sizeof(int));
    cudaMalloc(&d_hw_offsets, 33*sizeof(int));

    cudaMemcpy(d_chat, chat, TEST_TRACE*N*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strx, strx, 4*N*TEST_TRACE*5*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prob, prob, 4*N*TEST_TRACE*5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldrs, ldrs, 4*N*5*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_candidate, hw_candidate.data(),
               hw_candidate.size()*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_sizes, hw_sizes, 33*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_offsets, hw_offsets, 33*sizeof(int), cudaMemcpyHostToDevice);

    dfs_gpu_stats();

    return 0;
}
