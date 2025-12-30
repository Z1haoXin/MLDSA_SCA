#include "aes-independant.h"
#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "params.h"
#include "sign.h"
#include "fips202.h"
#include "symmetric.h"
#include "randombytes.h"

void decode(uint8_t *buf, int32_t* r){
	int64_t mask = 2+8*8380417;
	uint64_t n = 0;
	uint8_t i;
	for(i=0;i<8;i++){
		n = n*256 + buf[7-i];
	}
	n -= mask;
	*r = (int32_t)n;
}
void encode(uint8_t *buf, int32_t n){
	int64_t mask = 2+8*8380417;
	n += mask;
	uint8_t i;
	for(i=0;i<8;i++){
		buf[i] = n % 256;
		n /= 256;
	}
} 
uint8_t sign(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf){
    int i;
    uint8_t rhoprime[64];
    uint16_t nonce = 0;
    polyvecl s1, y, z;
    poly cp;
    uint8_t sig[32];

    srand((unsigned)time( NULL ) );
    randombytes(rhoprime, 64);
    PQCLEAN_MLDSA44_CLEAN_polyvecl_uniform_eta(&s1, rhoprime, 0);
    PQCLEAN_MLDSA44_CLEAN_polyvecl_ntt(&s1);
    
    randombytes(rhoprime, 64);
    PQCLEAN_MLDSA44_CLEAN_polyvecl_uniform_gamma1(&y, rhoprime, 0);
    
    randombytes(sig, 32);
    PQCLEAN_MLDSA44_CLEAN_poly_challenge(&cp, sig); 
    PQCLEAN_MLDSA44_CLEAN_poly_ntt(&cp);
    
    
    decode(buf, &cp.coeffs[0]);
    decode(buf+8, &s1.vec[0].coeffs[0]);
    trigger_high();
    PQCLEAN_MLDSA44_CLEAN_polyvecl_pointwise_poly_montgomery(&z, &cp, &s1);
    trigger_low();
    PQCLEAN_MLDSA44_CLEAN_polyvecl_invntt_tomont(&z);
    PQCLEAN_MLDSA44_CLEAN_polyvecl_add(&z, &z, &y);
    PQCLEAN_MLDSA44_CLEAN_polyvecl_reduce(&z);
    

    encode(buf, s1.vec[0].coeffs[0]);
    encode(buf+8, z.vec[0].coeffs[0]);
    simpleserial_put('r', 16, buf);
    return 0x00;
}
uint8_t pk[PQCLEAN_MLDSA44_CLEAN_CRYPTO_PUBLICKEYBYTES];
uint8_t sk[PQCLEAN_MLDSA44_CLEAN_CRYPTO_SECRETKEYBYTES];
int main(void){
    srand((unsigned)time(NULL));
    PQCLEAN_MLDSA44_CLEAN_crypto_sign_keypair(pk, sk);
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    simpleserial_addcmd(0x01, 16, sign);
    while(1)
        simpleserial_get();
}
