
/*
 * Built on cbuchner1's implementation, actual hashing code
 * based on sphlib 3.0
 *
 */
/*
 * Whirlpool kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
 *                     
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   djm34
 */

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

//#define PROF

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>
#ifdef DEBUG_HASH
#include <gmp.h>
#endif
#include <map>


#ifdef _WIN32 || _WIN64

#include <time.h>
#include <winsock2.h>
extern "C"{
int gettimeofday(struct timeval *tv, struct timezone *tz);
void usleep(int64_t waitTime);
}
typedef int64_t useconds_t;
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "uint256.h"
extern "C" {
#include "sph_whirlpool.h"
#include "sph_sha2.h"
#include "sph_tiger.h"
#include "sph_ripemd.h"
#include "sph_keccak.h"
#include "sph_haval.h"
}
#include "cuda_helper.h"
#include "trashminer.h"

extern void whirlpool512_cpu_init(int thr_id, int threads, int flag, ctx* pctx);
extern void sha256_cpu_init(int thr_id, int threads, ctx* pctx);
extern void sha512_cpu_init(int thr_id, int threads, ctx* pctx);
extern void haval256_cpu_init(int thr_id, int threads, ctx* pctx);
extern void tiger_cpu_init(int thr_id, int threads, ctx* pctx);
extern void ripemd_cpu_init(int thr_id, int threads, ctx* pctx);
extern void keccak512_cpu_init(int thr_id, int threads, ctx* pctx);

extern void sha256_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void sha512_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void haval256_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void tiger_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void ripemd_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void keccak512_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void whirlpool_scanhash(int throughput, uint64_t nonce, CBlockHeader *hdr, uint64_t *hash, ctx* pctx);
extern void sha256_fullhash(int throughput, uint64_t *data, uint64_t *hash);
extern void checkhash(int throughput, uint64_t *data, uint32_t *results, uint64_t target);

extern void cpu_mul(int order, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p);
extern void cpu_mulT4(int order, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p);
extern void mul_init();

// Zeitsynchronisations-Routine von cudaminer mit CPU sleep
typedef struct { double value[8]; } tsumarray;
cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id)
{
    cudaError_t result = cudaSuccess;
    if (situation >= 0)
    {   
        static std::map<int, tsumarray> tsum;

        double a = 0.95, b = 0.05;
        if (tsum.find(situation) == tsum.end()) { a = 0.5; b = 0.5; } // faster initial convergence

        double tsync = 0.0;
        double tsleep = 0.95 * tsum[situation].value[thr_id];
        if (cudaStreamQuery(stream) == cudaErrorNotReady)
        {
            usleep((useconds_t)(1e6*tsleep));
            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);
            result = cudaStreamSynchronize(stream);
            gettimeofday(&tv_end, NULL);
            tsync = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec);
        }
        if (tsync >= 0) tsum[situation].value[thr_id] = a * tsum[situation].value[thr_id] + b * (tsleep+tsync);
    }
    else
        result = cudaStreamSynchronize(stream);
    return result;
}

uint64_t swap_uint64( uint64_t val )
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL ) | ((val >> 8) & 0x00FF00FF00FF00FFULL );
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL ) | ((val >> 16) & 0x0000FFFF0000FFFFULL );
    return (val << 32) | (val >> 32);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const signed char p_util_hexdigit[256] =
{ -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,
  -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, };

const int throughput = 256*256*8;

#ifdef DEBUGHASH
void hash_cpu(CBlockHeader hdr){

	sph_sha256_context       ctx_sha256;
    	sph_sha512_context       ctx_sha512;
    	sph_keccak512_context    ctx_keccak;
    	sph_whirlpool_context    ctx_whirlpool;
    	sph_haval256_5_context   ctx_haval;
    	sph_tiger_context        ctx_tiger;
    	sph_ripemd160_context    ctx_ripemd;

	uint512 sph_hash[7];
	for(int i=0; i < 7; i++)
		sph_hash[i] = 0;


    	sph_sha256_init(&ctx_sha256);
    	// ZSHA256;
    	sph_sha256 (&ctx_sha256, &hdr, sizeof(hdr));
    	sph_sha256_close(&ctx_sha256, static_cast<void*>(&sph_hash[0]));
    
    	sph_sha512_init(&ctx_sha512);
    	// ZSHA512;
    	sph_sha512 (&ctx_sha512, &hdr, sizeof(hdr));
    	sph_sha512_close(&ctx_sha512, static_cast<void*>(&sph_hash[1]));
    
    	sph_keccak512_init(&ctx_keccak);
    	// ZKECCAK;
    	sph_keccak512 (&ctx_keccak, &hdr, sizeof(hdr));
    	sph_keccak512_close(&ctx_keccak, static_cast<void*>(&sph_hash[2]));

    	sph_whirlpool_init(&ctx_whirlpool);
    	// ZWHIRLPOOL;
    	sph_whirlpool (&ctx_whirlpool, &hdr, sizeof(hdr));
    	sph_whirlpool_close(&ctx_whirlpool, static_cast<void*>(&sph_hash[3]));
    
    	sph_haval256_5_init(&ctx_haval);
    	// ZHAVAL;
    	sph_haval256_5 (&ctx_haval, &hdr, sizeof(hdr));
    	sph_haval256_5_close(&ctx_haval, static_cast<void*>(&sph_hash[4]));

	sph_tiger_init(&ctx_tiger);
    	// ZTIGER;
    	sph_tiger (&ctx_tiger, &hdr, sizeof(hdr));
    	sph_tiger_close(&ctx_tiger, static_cast<void*>(&sph_hash[5]));

    	sph_ripemd160_init(&ctx_ripemd);
    	// ZRIPEMD;
    	sph_ripemd160 (&ctx_ripemd, &hdr, sizeof(hdr));
    	sph_ripemd160_close(&ctx_ripemd, static_cast<void*>(&sph_hash[6]));

//	for(int i=0; i < 7; i++)
//		printf("SPH: %s, L: %8.8X\n", sph_hash[i].GetHex().c_str(), ((uint32_t*)&sph_hash[i])[0]);

    	mpz_t bns[7];

    	//Take care of zeros and load gmp
    	for(int i=0; i < 7; i++){
	    if(sph_hash[i]==0)
	    	sph_hash[i] = 1;
	    mpz_init(bns[i]);
	    mpz_set_uint512(bns[i],sph_hash[i]);
        }

	mpz_t product;
    	mpz_init(product);
    	mpz_set_ui(product,1);
    	for(int i=0; i < 7; i++){
	 	mpz_mul(product,product,bns[i]);
    	}

	//gmp_printf ("Prod: %Zx\n", product);

        int bytes = mpz_sizeinbase(product, 256);
 	printf("Size in base: %d\n", bytes);

   	char *data = (char*)malloc(bytes);
    	mpz_export(data, NULL, -1, 1, 0, 0, product);

    	//Free the memory
    	for(int i=0; i < 7; i++){
		mpz_clear(bns[i]);
    	}
    	mpz_clear(product);

	uint256 finalhash=0;

    	sph_sha256_init(&ctx_sha256);
    	// ZSHA256;
    	sph_sha256 (&ctx_sha256, data,bytes);
    	sph_sha256_close(&ctx_sha256, static_cast<void*>(&finalhash));

	printf("Final hash: %s\n", finalhash.GetHex().c_str());
}

#endif

extern "C" {
uint64_t cuda_scanhash(void* ctx, void* data, void* target);
}

uint64_t cuda_scanhash(void *vctx, void* data, void* t){
	struct ctx *pctx = (struct ctx*)vctx;
	CBlockHeader hdr;
	uint256 target;
	memcpy(&hdr,data,sizeof(hdr));
	memcpy(&target,t,32);

	size_t hashSz =  8 * sizeof(uint64_t) * throughput;
	size_t prodSz = 38 * sizeof(uint64_t) * throughput;
	size_t resultsSz =  (throughput/512)*2048;

	//printf("Scanning block %ld %lX %s\n", hdr.nHeight, hdr.nNonce, target.GetHex().c_str());

	sha256_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[0], pctx);
	sha512_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[1], pctx);
	keccak512_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[2], pctx);
	whirlpool_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[3], pctx);
	haval256_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[4], pctx);
	tiger_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[5], pctx);
	ripemd_scanhash(throughput,hdr.nNonce,&hdr,pctx->d_hash[6], pctx);

	cpu_mulT4(0, throughput, 8, 8, pctx->d_hash[1], pctx->d_hash[2], pctx->d_prod[0]); //64
	MyStreamSynchronize(0,9,pctx->thr_id);

	cpu_mulT4(0, throughput,8, 16, pctx->d_hash[3], pctx->d_prod[0], pctx->d_prod[1]); //128
	MyStreamSynchronize(0,9,pctx->thr_id);

	cpu_mulT4(0, throughput, 4, 24, pctx->d_hash[0], pctx->d_prod[1], pctx->d_prod[0]); //96
	MyStreamSynchronize(0,10,pctx->thr_id);

	cpu_mulT4(0, throughput, 4, 28, pctx->d_hash[4], pctx->d_prod[0], pctx->d_prod[1]);  //112
	MyStreamSynchronize(0,11,pctx->thr_id);

	cpu_mul(0, throughput, 3, 32, pctx->d_hash[5], pctx->d_prod[1], pctx->d_prod[0]); //96
	MyStreamSynchronize(0,12,pctx->thr_id);

	cpu_mul(0, throughput, 3, 35, pctx->d_hash[6], pctx->d_prod[0], pctx->d_prod[1]); //105
	MyStreamSynchronize(0,13,pctx->thr_id);

	sha256_fullhash(throughput,pctx->d_prod[1],pctx->d_hash[7]);

	uint64_t startNonce = hdr.nNonce;

	//Check for any winners
	uint64_t targetword = ((uint64_t)(((uint32_t*)&target)[7]) << 32) | ((uint32_t*)&target)[6];

	//printf("%16.16lX\n", targetword);

	checkhash(throughput,pctx->d_hash[7],pctx->d_results, targetword);

//	cudaMemcpyAsync( pctx->hash[7], pctx->d_hash[7], hashSz/2, cudaMemcpyDeviceToHost, 0 );
	cudaMemcpyAsync( pctx->results, pctx->d_results, resultsSz, cudaMemcpyDeviceToHost, 0 ); 

	MyStreamSynchronize(0,15,pctx->thr_id);

	for(int i=0; i < throughput; i++){
		//First locate block
		int block = i / 512; 
		//Start offset is block * 4096
		int sofst = block * 4096/32;
		int thread = i % 512;
		int word = thread / 32;
		
		uint32_t set = pctx->results[sofst + word];
		uint32_t r = (set >> (thread%32)) & 1;
		if(r){
			printf("Checkhash found a winner, nonce %ld\n", startNonce + i* 0x100000000ULL); 
			hdr.nNonce = startNonce+i* 0x100000000ULL;
#ifdef PROF
cudaDeviceReset();
exit(0);
#endif
			return hdr.nNonce;
		}
	}


	return 0;	
	for(int i=0; i < throughput; i++){
		//Only really need to check high word
		uint64_t highword = pctx->hash[7][3*throughput+i];
		if(highword < targetword){
			printf("Found a winner, %lX nonce %ld\n", highword, startNonce + i* 0x100000000ULL); 
			hdr.nNonce = startNonce+i* 0x100000000ULL;
#ifdef DEBUG_HASH
			hash_cpu(hdr);
#endif
			return hdr.nNonce;
		}		
	}



	//printf("Done scan\n");
	return 0;
} 

#if 0
void hash_gpu(int throughput, CBlockHeader hdr){

	sha256_scanhash(throughput,hdr.nNonce,&hdr,d_hash[0]);
	sha512_scanhash(throughput,hdr.nNonce,&hdr,d_hash[1]);
	keccak512_scanhash(throughput,hdr.nNonce,&hdr,d_hash[2]);
	whirlpool_scanhash(throughput,hdr.nNonce,&hdr,d_hash[3]);
	haval256_scanhash(throughput,hdr.nNonce,&hdr,d_hash[4]);
	tiger_scanhash(throughput,hdr.nNonce,&hdr,d_hash[5]);
	ripemd_scanhash(throughput,hdr.nNonce,&hdr,d_hash[6]);

#if 0
	cpu_mul(0, throughput, 4, 8, d_hash[0], d_hash[1], d_prod[0]);
	cpu_mul(0, throughput, 8, 12, d_hash[2], d_prod[0], d_prod[1]);
	cpu_mul(0, throughput, 8, 20, d_hash[3], d_prod[1], d_prod[0]);
	cpu_mul(0, throughput, 4, 28, d_hash[4], d_prod[0], d_prod[1]);
	cpu_mul(0, throughput, 3, 32, d_hash[5], d_prod[1], d_prod[0]);
	cpu_mul(0, throughput, 3, 35, d_hash[6], d_prod[0], d_prod[1]);
#else
	cpu_mul(0, throughput, 3, 3, d_hash[6], d_hash[5], d_prod[0]);
	cpu_mul(0, throughput, 4, 6, d_hash[0], d_prod[0], d_prod[1]);
	cpu_mul(0, throughput, 4, 10, d_hash[4], d_prod[1], d_prod[0]);
	cpu_mul(0, throughput, 8, 14, d_hash[3], d_prod[0], d_prod[1]);
	cpu_mul(0, throughput, 8, 22, d_hash[2], d_prod[1], d_prod[0]);
	cpu_mul(0, throughput, 8, 30, d_hash[1], d_prod[0], d_prod[1]);

#endif
	sha256_fullhash(throughput,d_prod[1],d_hash[7]);

#if 0
	for(int i=0; i < 8; i++){
		cudaMemcpy( hash[i], d_hash[i], hashSz, cudaMemcpyDeviceToHost ); 

		for(int j=0; j < 8; j++)
			printf("%16.16lx",hash[i][(7-j)*throughput]);
		printf("\n");

		cudaFree(d_hash[i]);
		free(hash[i]);
	}

	for(int i=1; i < 2; i++){
		cudaMemcpy( prod[i], d_prod[i], prodSz, cudaMemcpyDeviceToHost ); 

		for(int j=0; j < 38; j++)
			printf("%16.16lx",prod[i][(37-j)*throughput]);
		printf("\n");

		cudaFree(d_prod[i]);
		free(prod[i]);
	}
#endif

}


int main2(){
	whirlpool512_cpu_init(0,throughput,0);
	sha256_cpu_init(0,throughput);
	sha512_cpu_init(0,throughput);
	haval256_cpu_init(0,throughput);
	tiger_cpu_init(0,throughput);
	ripemd_cpu_init(0,throughput);
	keccak512_cpu_init(0,throughput);

	size_t hashSz =  8 * sizeof(uint64_t) * throughput;
	size_t prodSz = 38 * sizeof(uint64_t) * throughput;
	for(int i=0; i < 8; i++){
		gpuErrchk(cudaMalloc(&d_hash[i], hashSz));
		hash[i] = (uint64_t*)malloc(hashSz);
	}

	for(int i=0; i < 2; i++){
		gpuErrchk(cudaMalloc(&d_prod[i], prodSz));
		prod[i] = (uint64_t*)malloc(prodSz);
	}


	CBlockHeader hdr;
	hdr.hashPrevBlock = uint256("0000000006c61169d8fa4aa773858bd39988de2f4fecea2ee51f1606bb87e6f6");
	hdr.hashMerkleRoot = uint256("f05690e81cae7a8aacf81408a107d26c271e29303547407d672049674069ba15");
	hdr.hashAccountRoot = uint256("e5922c85dc791c5ec17f923d26cd78c6ee3235f88585bcc3779ef7fab61dbf8c");
	hdr.nNonce = 12552329922997286027ULL;	
	hdr.nTime = 1406843021;
	hdr.nVersion = 1;
	hdr.nHeight = 7000;

	printf("%ld\n", sizeof(hdr));

	hash_cpu(hdr);
	hash_gpu(throughput,hdr);

}
#endif


extern "C" {
	void* cuda_init(int id);
}

void* cuda_init(int id){
	const int throughput = 256*256*8;
	gpuErrchk(cudaSetDevice(id));

	ctx *pctx = new ctx;
	pctx->thr_id = 0;

	cudaStreamCreate(&pctx->stream);

	whirlpool512_cpu_init(0,throughput,0, pctx);
	sha256_cpu_init(0,throughput,pctx);
	sha512_cpu_init(0,throughput, pctx);
	haval256_cpu_init(0,throughput,pctx);
	tiger_cpu_init(0,throughput, pctx);
	ripemd_cpu_init(0,throughput, pctx);
	keccak512_cpu_init(0,throughput,pctx);
	mul_init();

	size_t hashSz =  8 * sizeof(uint64_t) * throughput;
	size_t prodSz = 38 * sizeof(uint64_t) * throughput;
	for(int i=0; i < 8; i++){
		gpuErrchk(cudaMalloc(&pctx->d_hash[i], hashSz));
		pctx->hash[i] = (uint64_t*)malloc(hashSz);
	}

	for(int i=0; i < 2; i++){
		gpuErrchk(cudaMalloc(&pctx->d_prod[i], prodSz));
		pctx->prod[i] = (uint64_t*)malloc(prodSz);
	}

	//Results are spaced out so no conflicts on global memory. cache line is 256bytes
	size_t resultsSz =  (throughput/512)*2048;

	pctx->results = (uint32_t*)malloc(resultsSz);
	gpuErrchk(cudaMalloc(&pctx->d_results, resultsSz));

	return pctx;
}

