#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "params.h"
#include "elmoasmfunctionsdef-extension.h"
#include "reduce.h"

void decode(uint8_t *buf, int32_t* r);
void encode(uint8_t *buf, int32_t n);

// ELMO API :
//  - printbyte(addr): Print single byte located at address 'addr' to output file;
//  - randbyte(addr): Load byte of random to memory address 'addr';
//  - readbyte(addr): Read byte from input file to address 'addr'.
// ELMO API (extension) :
//  - print2bytes, rand2bytes and read2bytes: idem, but for an address pointing on 2 bytes;
//  - print4bytes, rand4bytes and read4bytes: idem, but for an address pointing on 4 bytes.
void decode(uint8_t *buf, int32_t* r){
	int64_t mask = 2+8*Q;
	uint64_t n = 0;
	uint8_t i;
	for(i=0;i<8;i++){
		n = n*256 + buf[7-i];
	}
	n -= mask;
	*r = (int32_t)n;
}
void encode(uint8_t *buf, int32_t n){
	int64_t mask = 2+8*Q;
	n += mask;
	uint8_t i;
	for(i=0;i<8;i++){
		buf[i] = n % 256;
		n /= 256;
	}
} 

int main(void)
{
	uint16_t num_challenge, nb_challenges;
	int32_t s1[2], c[2], x[2];
	uint8_t buf[8];

	read2bytes(&nb_challenges);
	
	for (num_challenge = 0; num_challenge < nb_challenges; num_challenge++) {
		for (int i = 0; i < 1; i++) {
			for(int h=0;h<8;h++) readbyte(&(buf[h]));
			decode(buf, &c[i]);
		}
		for (int i = 0; i < 1; i++) {
			for(int h=0;h<8;h++) readbyte(&(buf[h]));
			decode(buf, &s1[i]);
		}
		
		starttrigger(); // To start a new trace
		// Do the leaking operations here...
		// for (int i = 0; i < 2; ++i) {
        // 	x[i] = PQCLEAN_MLDSA44_CLEAN_montgomery_reduce((int64_t)c[i] * s1[i]);
    	// }
		x[0] = PQCLEAN_MLDSA44_CLEAN_montgomery_reduce((int64_t)c[0] * s1[0]);
		x[1] = PQCLEAN_MLDSA44_CLEAN_montgomery_reduce((int64_t)c[1] * s1[1]);
		endtrigger(); // To end the current trace

	}
	endprogram(); // To indicate to ELMO that the simulation is finished

	return 0;
}
