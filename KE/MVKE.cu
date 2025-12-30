#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/time.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <hdf5/serial/hdf5.h>

#define Q 8380417
#define QINV 58728449
#define N 256
#define TEST_TRACE 3000
#define TRACE_TILE 64 
#define TOTAL_COEFF (4*N)

using namespace std;

int base_eqs, trc_num;
int32_t chat[TEST_TRACE][N];
int32_t shat[4][N];
int32_t ldrs[4][N][5];
int32_t strx[4][N][TEST_TRACE][5];

vector<int32_t> hw_candidate; 
int hw_sizes[33];
int hw_offsets[33];

int32_t *d_chat;
int32_t *d_strx;
int32_t *d_ldrs;
int32_t *d_candidate;
int *d_hit;
int32_t *d_hw_candidate;
int *d_hw_sizes;
int *d_hw_offsets;


__host__ __device__ int32_t montgomery_reduce(int64_t a) {
    int32_t t = (int32_t)a * QINV;
    return (a - (int64_t)t * Q) >> 32;
}


bool load_data() {
    hid_t file_id, dataset_id;
    herr_t status;
    file_id = H5Fopen("../TA/Trace/attack_data_cs1.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) { printf("fail to open file\n"); return false; }
    dataset_id = H5Dopen2(file_id, "/chat", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, chat);
    H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "/s1hat", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shat);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return (status >= 0);
}

bool load_hw() {
    hid_t file_id, dataset_id;
    herr_t status;
    for(int q=0;q<4;q++){
        char filename[60];
        sprintf(filename, "../TA/match_res/match_res_shat_rk5_no%d.h5", q);
        file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) { printf("fail to open file\n"); return false; }
        dataset_id = H5Dopen2(file_id, "/hw", H5P_DEFAULT);
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ldrs[q]);
        H5Dclose(dataset_id);
        H5Fclose(file_id);

        sprintf(filename, "../TA/match_res/match_res_xhat_rk5_no%d.h5", q);
        file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) { printf("fail to open file\n"); return false; }
        for(int idx=0; idx<N; idx++){
            char index[10];
            sprintf(index, "/%d/hw", idx);
            dataset_id = H5Dopen2(file_id, index, H5P_DEFAULT);
            status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, strx[q][idx]);
            H5Dclose(dataset_id);
        }
        H5Fclose(file_id);
    }
    return (status >= 0);
}

void init_hw_candidate() {
    memset(hw_sizes, 0, sizeof(hw_sizes));
    for (int32_t a=-2-4*Q; a<=2+4*Q; a++)
        hw_sizes[__builtin_popcount(a)]++;

    hw_offsets[0]=0;
    for(int hw=1; hw<=32; hw++)
        hw_offsets[hw]=hw_offsets[hw-1]+hw_sizes[hw-1];

    hw_candidate.resize(hw_offsets[32]+hw_sizes[32]);
    int cnt[33]={0};
    for(int32_t a=-2-4*Q; a<=2+4*Q; a++){
        int hw = __builtin_popcount(a);
        hw_candidate[hw_offsets[hw]+cnt[hw]++] = a;
    }
}

void init_gpu_memory() {
    cudaMalloc(&d_chat, TEST_TRACE*N*sizeof(int32_t));
    cudaMalloc(&d_strx, 4*N*TEST_TRACE*5*sizeof(int32_t));
    cudaMalloc(&d_ldrs, 4*N*5*sizeof(int32_t));
    cudaMalloc(&d_candidate, 4*N*sizeof(int32_t));
    cudaMalloc(&d_hit, 4*N*sizeof(int));
    cudaMalloc(&d_hw_candidate, hw_candidate.size()*sizeof(int32_t));
    cudaMalloc(&d_hw_sizes, 33*sizeof(int));
    cudaMalloc(&d_hw_offsets, 33*sizeof(int));

    cudaMemcpy(d_chat, chat, TEST_TRACE*N*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strx, strx, 4*N*TEST_TRACE*5*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldrs, ldrs, 4*N*5*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_candidate, hw_candidate.data(), hw_candidate.size()*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_sizes, hw_sizes, 33*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw_offsets, hw_offsets, 33*sizeof(int), cudaMemcpyHostToDevice);
}

void cleanup_gpu_memory() {
    cudaFree(d_chat);
    cudaFree(d_strx);
    cudaFree(d_ldrs);
    cudaFree(d_candidate);
    cudaFree(d_hit);
    cudaFree(d_hw_candidate);
    cudaFree(d_hw_sizes);
    cudaFree(d_hw_offsets);
}

__global__ void majority_vote_kernel_hw(
    const int32_t* __restrict__ d_chat,
    const int32_t* __restrict__ d_strx,
    const int32_t* __restrict__ d_ldrs,
    int trace_num,
    int min_eqs,
    const int32_t* __restrict__ d_hw_candidate,
    const int* __restrict__ d_hw_sizes,
    const int* __restrict__ d_hw_offsets,
    int32_t* d_candidate,
    int* d_hit
){
    extern __shared__ int32_t shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int idx = bid & 255;
    int q   = bid >> 8;
    int global_idx = (q<<8)+idx;

    int base_ldrs = q*N*5+idx*5;
    int l0=d_ldrs[base_ldrs+0], l1=d_ldrs[base_ldrs+1], l2=d_ldrs[base_ldrs+2],
        l3=d_ldrs[base_ldrs+3], l4=d_ldrs[base_ldrs+4];

    int local_hit=0;
    int32_t local_cand=0;

    int hws[5]={l0,l1,l2,l3,l4};
    for(int i=0;i<5;i++){
        int hw = hws[i];
        int start = d_hw_offsets[hw];
        int len   = d_hw_sizes[hw];

        for(int j=tid;j<len;j+=blockDim.x){
            int32_t a = d_hw_candidate[start+j];
            int vote=0;
            int base_strx = (q*N+idx)*TEST_TRACE*5;
            for(int t=min_eqs;t<min_eqs+trace_num;t++){
                int32_t hwx = __popc((uint32_t)montgomery_reduce((int64_t)d_chat[t*N+idx]*a));
                int off = base_strx + t*5;
                if(hwx==d_strx[off+0]||hwx==d_strx[off+1]||hwx==d_strx[off+2]||
                   hwx==d_strx[off+3]||hwx==d_strx[off+4])
                    vote++;
            }
            if(vote>local_hit || (vote==local_hit && a<local_cand)){
                local_hit=vote;
                local_cand=a;
            }
        }
    }

    shared[tid]=local_hit;
    shared[blockDim.x+tid]=local_cand;
    __syncthreads();

    if(tid==0){
        int block_max=0;
        int32_t block_cand=0;
        for(int i=0;i<blockDim.x;i++){
            int vote_i=shared[i];
            int32_t cand_i=shared[blockDim.x+i];
            if(vote_i>block_max||(vote_i==block_max && cand_i<block_cand)){
                block_max=vote_i;
                block_cand=cand_i;
            }
        }
        d_hit[global_idx]=block_max;
        d_candidate[global_idx]=block_cand;
    }
}

void majority_vote_gpu(int trace_num){
    int threads_per_block=256;
    int num_blocks=TOTAL_COEFF; 

    size_t shared_mem = threads_per_block*2*sizeof(int32_t);
    cudaMemset(d_hit,0,TOTAL_COEFF*sizeof(int));
    cudaMemset(d_candidate,0,TOTAL_COEFF*sizeof(int32_t));

    majority_vote_kernel_hw<<<num_blocks, threads_per_block, shared_mem>>>(
        d_chat, d_strx, d_ldrs, trace_num, base_eqs,
        d_hw_candidate, d_hw_sizes, d_hw_offsets,
        d_candidate, d_hit
    );

    cudaDeviceSynchronize();
}

void dfs_gpu_stats(){
    int32_t h_candidate[4*N];
    int h_hit[4*N];
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    majority_vote_gpu(trc_num);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("majority_vote_gpu time: %.2f ms\n", elapsed);

    cudaMemcpy(h_candidate, d_candidate, 4*N*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hit, d_hit, 4*N*sizeof(int), cudaMemcpyDeviceToHost);

    int recovered=0;
    for(int idx=0;idx<4*N;idx++){
        int q_idx = idx/256;
        int n_idx = idx%256;
        if(h_candidate[idx]==shat[q_idx][n_idx] ||
            montgomery_reduce((int64_t)h_candidate[idx])==montgomery_reduce((int64_t)shat[q_idx][n_idx]))
            recovered++;
    }
    printf("#trace_num=%d recovered=%d / %d\n",trc_num,recovered,4*N);
    fflush(stdout);
}

int main(int argc, char* argv[]){
    if(argc<3){ cout<<"usage: "<<argv[0]<<" <base_eqs> <trc_num>\n"; return 1; }
    base_eqs = atoi(argv[1]);
    trc_num = atoi(argv[2]);

    load_data();
    load_hw();
    init_hw_candidate();
    init_gpu_memory();
    dfs_gpu_stats();
    cleanup_gpu_memory();
    return 0;
}
