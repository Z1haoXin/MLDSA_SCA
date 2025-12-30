#include<bits/stdc++.h>
#include<sys/time.h>
#include <hdf5/serial/hdf5.h>
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define Q 8380417
#define N 256
#define V 9
#define MAX_LINE_LENGTH 1000
#define Search_BITS 64
using namespace std;

static const int32_t zetas[N] = {
         0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
   1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
   2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
  -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
   2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
  -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
  -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
    811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
  -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
  -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
   3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
   -671102, -1228525,   -22981, -1308169,  -381987,  1349076,  1852771, -1430430,
  -3343383,   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618,
  -3724342,    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,
    189548, -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,
   1285669, -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,
   2091667,  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,
    266997,  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,
    900702,  1859098,   909542,   819034,   495491, -1613174,   -43260,  -522500,
   -655327, -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,
    342297,   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,
   2842341,  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974,
  -3767016,  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970,
  -1333058,  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642,
  -1279661,  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,
   -542412, -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993,
  -2013608,  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385,
  -3183426,   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107,
  -3038916,  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,
   -426683,  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893,
  -2939036, -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,
   -554416,  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};

int32_t montgomery_reduce(int64_t a){
  int32_t t;
  t = (int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;
}

void ntt(int32_t a[N]) {
  unsigned int len, start, j, k;
  int32_t zeta, t;
  k = 0;
  for(len = 128; len > 0; len >>= 1) {
    for(start = 0; start < N; start = j + len) {
      zeta = zetas[++k];
      for(j = start; j < start + len; ++j) {
        t = montgomery_reduce((int64_t)zeta * a[j + len]);
        a[j + len] = a[j] - t;
        a[j] = a[j] + t;
      }
    }
  }
}

int hit_rate;
int32_t cp[N];
int32_t ct[N];
int32_t hwset[N][5];
vector<int> POS;
int32_t c[N];

int TMP[1000][5] = {0};
bool readH5(const char *filepath) {
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    
    file_id = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("fail to open file\n");
        return false;
    }
    
    dataset_id = H5Dopen2(file_id, "/data", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, TMP);
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    
    return (status >= 0);
}


void loadhw(int q){
	for(int i=0;i<N;i++){
		char filename[40];
		sprintf(filename, "../TA/res/match_res_c_idx_%d.h5", i);
		readH5(filename);
		for(int j=0;j<5;j++) hwset[i][j] = TMP[q][j];
	}
}

void decodec(int32_t c[N], uint8_t *buf){
    for(int i=0;i<64;i++){
        c[4*i] = buf[i] & 0x3;
        c[4*i+1] = (buf[i]>>2) & 0x3;
        c[4*i+2] = (buf[i]>>4) & 0x3;
        c[4*i+3] = (buf[i]>>6) & 0x3;
    }
    for(int i=0;i<N;i++) c[i]=c[i]-1;
} 


bool loadcp(int q) {
	uint8_t TMP_cp[1000][64];
	hid_t file_id, dataset_id;
	herr_t status;
	file_id = H5Fopen("../TA/trace/attack_data_challenge.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) { printf("fail to open file\n"); return false; }
	dataset_id = H5Dopen2(file_id, "/data", H5P_DEFAULT);
	status = H5Dread(dataset_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, TMP_cp);
	H5Dclose(dataset_id);
	H5Fclose(file_id);

	decodec(cp, TMP_cp[q]);

	return (status >= 0);
}

long long C(int n, int m) {
	if (m < n - m) m = n - m;
	long long ans = 1;
	for (int i = m + 1; i <= n; i++) ans *= i;
	for (int i = 1; i <= n - m; i++) ans /= i;
	return ans;
}


void FIX(){
	int32_t ctcp[N];
	memcpy(ctcp, ct, sizeof(ct));
	ctcp[0] = 1-ctcp[0];
	ntt(ctcp);
	int hr = 0;
	for(int i=0;i<N;i++){
		int32_t hw = __builtin_popcount(ctcp[i]);
		if(hw==hwset[i][0] || hw==hwset[i][1] || hw==hwset[i][2] || hw==hwset[i][3] || hw==hwset[i][4]) hr++;
	}
	if(hr > hit_rate) hit_rate = hr, ct[0] = 1-ct[0];
}

void load(int q){
  	loadcp(q);
	loadhw(q);
	memset(ct, 0, sizeof(ct));
	for(int i=0;i<Search_BITS;i++){
		if(cp[i]==-1) ct[i] = cp[i];
		else if(cp[i]!=-1 && i>=32){
			int32_t p = cp[64+i] + montgomery_reduce((int64_t)cp[192+i]*25847);
			if(cp[128+i] == 0 && p == 0) ct[i] = cp[i];
			else POS.push_back(i);
		}
		else POS.push_back(i);
	} 
	for(int i=Search_BITS;i<N;i++) ct[i] = cp[i];
}

void add_noise(double th){
    int32_t ctcp[N];

	memcpy(ctcp, cp, sizeof(cp));
	ntt(ctcp);
	double sr = 0;
	for(int j=0;j<N;j++){
		int32_t hw = __builtin_popcount(ctcp[j]);
		for(int k=0;k<5;k++) if(hw == hwset[j][k]) sr++;
	}
	printf("sr = %.2lf\n", sr/256.0);
	while(sr/256.0 > th){
		int idx = rand()%256;
		int32_t hw = __builtin_popcount(ctcp[idx]);
		for(int k=0;k<5;k++) if(hw == hwset[idx][k]) hwset[idx][k] = rand()%33;
		sr = 0;
		for(int j=0;j<N;j++){
			int32_t hw = __builtin_popcount(ctcp[j]);
			for(int k=0;k<5;k++) if(hw == hwset[j][k]) sr++;
		}
	}
    
	sr = 0;
	for(int j=0;j<N;j++){
		int32_t hw = __builtin_popcount(ctcp[j]);
		for(int k=0;k<5;k++) if(hw == hwset[j][k]) sr++;
	}
	printf("sr = %.2lf\n", sr/256.0);
    
}


int main(){
	for(int q=0;q<1000;q++){
		printf("=== NO.%d ===\n", q);
		hit_rate = 0;
		POS.clear();

		load(q);

		struct timeval begin,end;
		double avetime=0;
		int sr = 0;

		int cnt01 = 0, cnt1 = 0;
		for(auto pos: POS){cnt01++; if(cp[pos]==1) cnt1++;} 
		cout<<"No."<<q<<" "<<log2(C(cnt01, cnt1))<<endl;
		if(log2(C(cnt01, cnt1)) >= 24) continue;

		vector<int> B(cnt01, 0), A(cnt01, 0);
		fill(B.end()-cnt1, B.end(), 1);


		gettimeofday(&begin, 0);
		do{
			memcpy(c, ct, sizeof(ct));
			for(int i=0;i<POS.size();i++) c[POS[i]] = (int32_t)B[i];
			ntt(c);
			int hr = 0;
			for(int i=0;i<N;i++){
				int32_t hw = __builtin_popcount(c[i]);
				if(hw==hwset[i][0] || hw==hwset[i][1] || hw==hwset[i][2] || hw==hwset[i][3] || hw==hwset[i][4]) hr++;
			}
			if(hr > hit_rate) {hit_rate=hr; A = B;}
		}while(next_permutation(B.begin(), B.end()));
		gettimeofday(&end, 0);

		long sec = end.tv_sec - begin.tv_sec;
		long microsec = end.tv_usec - begin.tv_usec;
		double elapsed = sec + microsec*1e-6;
		for(int i=0;i<POS.size();i++) ct[POS[i]] = (int32_t)A[i];

		if(ct[0] == 0 || ct[0] == 1) FIX();
		bool f = false;
		if(memcmp(ct, cp, sizeof(ct)) == 0){
			avetime += elapsed;
			sr++;
			f = true;
		}
		else printf("No.%d: fail to recover\n", q);

		printf("No.%d success: hit rate = %d time = %.4lfs\n", q, hit_rate, elapsed);
		for(int i=0;i<N;i++) printf("%d", ct[i]+1); printf("\n");
		for(int i=0;i<N;i++) printf("%d", cp[i]+1); printf("\n");

		printf("done.\n");
	}
	
    return 0;
}
// complie with 
// g++ -o3 bn_search.cpp -o bn_search -lhdf5_serial