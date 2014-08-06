/*
 * sha256 djm34, catia
 * 
 */

/*
 * sha-256 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
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
 * @author   phm <phm@inbox.com>
 */

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>
#include "uint256.h"
extern "C" {
#include "sph_sha2.h"
}
#define USE_SHARED 1
#include "cuda_helper.h"

#include "trashminer.h"

#define SPH_C32(x)    ((uint64_t)(x ## U))
//#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#define SPH_T32(x)  sph_t32(x)

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

typedef uint32_t sph_t32;
typedef uint32_t sph_u32;


#if 1
static __device__ __forceinline__ uint32_t SWAP32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}
#endif

__device__ __forceinline__ uint64_t SWAP32IN64(uint64_t x)
{
	// Input:	77665544 33221100
	// Output:	00112233 44556677
	uint64_t temp[2];
	temp[0] = __byte_perm(HIWORD(x), 0, 0x0123);
	temp[1] = __byte_perm(LOWORD(x), 0, 0x0123);

	return (temp[0] << 0) | (temp[1] << 32);
}

static __host__ uint32_t SWAP(uint32_t val){
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF ); 
    return (val << 16) | (val >> 16);
                 
}

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

static __constant__ uint32_t H_256[8];

static const uint32_t H256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85), SPH_C32(0x3C6EF372),
	SPH_C32(0xA54FF53A), SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};

static __constant__ uint32_t K_256[80];

static const uint32_t K256[80] = {
	SPH_C32(0x428A2F98), SPH_C32(0x71374491),
	SPH_C32(0xB5C0FBCF), SPH_C32(0xE9B5DBA5),
	SPH_C32(0x3956C25B), SPH_C32(0x59F111F1),
	SPH_C32(0x923F82A4), SPH_C32(0xAB1C5ED5),
	SPH_C32(0xD807AA98), SPH_C32(0x12835B01),
	SPH_C32(0x243185BE), SPH_C32(0x550C7DC3),
	SPH_C32(0x72BE5D74), SPH_C32(0x80DEB1FE),
	SPH_C32(0x9BDC06A7), SPH_C32(0xC19BF174),
	SPH_C32(0xE49B69C1), SPH_C32(0xEFBE4786),
	SPH_C32(0x0FC19DC6), SPH_C32(0x240CA1CC),
	SPH_C32(0x2DE92C6F), SPH_C32(0x4A7484AA),
	SPH_C32(0x5CB0A9DC), SPH_C32(0x76F988DA),
	SPH_C32(0x983E5152), SPH_C32(0xA831C66D),
	SPH_C32(0xB00327C8), SPH_C32(0xBF597FC7),
	SPH_C32(0xC6E00BF3), SPH_C32(0xD5A79147),
	SPH_C32(0x06CA6351), SPH_C32(0x14292967),
	SPH_C32(0x27B70A85), SPH_C32(0x2E1B2138),
	SPH_C32(0x4D2C6DFC), SPH_C32(0x53380D13),
	SPH_C32(0x650A7354), SPH_C32(0x766A0ABB),
	SPH_C32(0x81C2C92E), SPH_C32(0x92722C85),
	SPH_C32(0xA2BFE8A1), SPH_C32(0xA81A664B),
	SPH_C32(0xC24B8B70), SPH_C32(0xC76C51A3),
	SPH_C32(0xD192E819), SPH_C32(0xD6990624),
	SPH_C32(0xF40E3585), SPH_C32(0x106AA070),
	SPH_C32(0x19A4C116), SPH_C32(0x1E376C08),
	SPH_C32(0x2748774C), SPH_C32(0x34B0BCB5),
	SPH_C32(0x391C0CB3), SPH_C32(0x4ED8AA4A),
	SPH_C32(0x5B9CCA4F), SPH_C32(0x682E6FF3),
	SPH_C32(0x748F82EE), SPH_C32(0x78A5636F),
	SPH_C32(0x84C87814), SPH_C32(0x8CC70208),
	SPH_C32(0x90BEFFFA), SPH_C32(0xA4506CEB),
	SPH_C32(0xBEF9A3F7), SPH_C32(0xC67178F2)
};

#define SHA2_MEXP1(in, pc)   do { \
		W[pc] = in(pc); \
	} while (0)

#define SHA2_MEXP2(in, pc)   do { \
		W[(pc) & 0x0F] = SPH_T32(SSG2_1(W[((pc) - 2) & 0x0F]) \
			+ W[((pc) - 7) & 0x0F] \
			+ SSG2_0(W[((pc) - 15) & 0x0F]) + W[(pc) & 0x0F]); \
	} while (0)

#define SHA2_STEPn(n, a, b, c, d, e, f, g, h, in, pc)   do { \
		sph_u32 t1, t2; \
		SHA2_MEXP ## n(in, pc); \
		t1 = SPH_T32(h + BSG2_1(e) + CH(e, f, g) \
			+ K_256[pcount + (pc)] + W[(pc) & 0x0F]); \
		t2 = SPH_T32(BSG2_0(a) + MAJ(a, b, c)); \
		d = SPH_T32(d + t1); \
		h = SPH_T32(t1 + t2); \
	} while (0)

#define SHA2_STEP1(a, b, c, d, e, f, g, h, in, pc) \
	SHA2_STEPn(1, a, b, c, d, e, f, g, h, in, pc)
#define SHA2_STEP2(a, b, c, d, e, f, g, h, in, pc) \
	SHA2_STEPn(2, a, b, c, d, e, f, g, h, in, pc)

#define SHA2_ROUND_BODY(in, r)   do { \
		sph_u32 A, B, C, D, E, F, G, H; \
		sph_u32 W[16]; \
		unsigned pcount; \
 \
		A = (r)[0]; \
		B = (r)[1]; \
		C = (r)[2]; \
		D = (r)[3]; \
		E = (r)[4]; \
		F = (r)[5]; \
		G = (r)[6]; \
		H = (r)[7]; \
		pcount = 0; \
		SHA2_STEP1(A, B, C, D, E, F, G, H, in,  0); \
		SHA2_STEP1(H, A, B, C, D, E, F, G, in,  1); \
		SHA2_STEP1(G, H, A, B, C, D, E, F, in,  2); \
		SHA2_STEP1(F, G, H, A, B, C, D, E, in,  3); \
		SHA2_STEP1(E, F, G, H, A, B, C, D, in,  4); \
		SHA2_STEP1(D, E, F, G, H, A, B, C, in,  5); \
		SHA2_STEP1(C, D, E, F, G, H, A, B, in,  6); \
		SHA2_STEP1(B, C, D, E, F, G, H, A, in,  7); \
		SHA2_STEP1(A, B, C, D, E, F, G, H, in,  8); \
		SHA2_STEP1(H, A, B, C, D, E, F, G, in,  9); \
		SHA2_STEP1(G, H, A, B, C, D, E, F, in, 10); \
		SHA2_STEP1(F, G, H, A, B, C, D, E, in, 11); \
		SHA2_STEP1(E, F, G, H, A, B, C, D, in, 12); \
		SHA2_STEP1(D, E, F, G, H, A, B, C, in, 13); \
		SHA2_STEP1(C, D, E, F, G, H, A, B, in, 14); \
		SHA2_STEP1(B, C, D, E, F, G, H, A, in, 15); \
		for (pcount = 16; pcount < 64; pcount += 16) { \
			SHA2_STEP2(A, B, C, D, E, F, G, H, in,  0); \
			SHA2_STEP2(H, A, B, C, D, E, F, G, in,  1); \
			SHA2_STEP2(G, H, A, B, C, D, E, F, in,  2); \
			SHA2_STEP2(F, G, H, A, B, C, D, E, in,  3); \
			SHA2_STEP2(E, F, G, H, A, B, C, D, in,  4); \
			SHA2_STEP2(D, E, F, G, H, A, B, C, in,  5); \
			SHA2_STEP2(C, D, E, F, G, H, A, B, in,  6); \
			SHA2_STEP2(B, C, D, E, F, G, H, A, in,  7); \
			SHA2_STEP2(A, B, C, D, E, F, G, H, in,  8); \
			SHA2_STEP2(H, A, B, C, D, E, F, G, in,  9); \
			SHA2_STEP2(G, H, A, B, C, D, E, F, in, 10); \
			SHA2_STEP2(F, G, H, A, B, C, D, E, in, 11); \
			SHA2_STEP2(E, F, G, H, A, B, C, D, in, 12); \
			SHA2_STEP2(D, E, F, G, H, A, B, C, in, 13); \
			SHA2_STEP2(C, D, E, F, G, H, A, B, in, 14); \
			SHA2_STEP2(B, C, D, E, F, G, H, A, in, 15); \
		} \
		(r)[0] = SPH_T32((r)[0] + A); \
		(r)[1] = SPH_T32((r)[1] + B); \
		(r)[2] = SPH_T32((r)[2] + C); \
		(r)[3] = SPH_T32((r)[3] + D); \
		(r)[4] = SPH_T32((r)[4] + E); \
		(r)[5] = SPH_T32((r)[5] + F); \
		(r)[6] = SPH_T32((r)[6] + G); \
		(r)[7] = SPH_T32((r)[7] + H); \
	} while (0)


#define BSG2_0(x)      (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define BSG2_1(x)      (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SSG2_0(x)      (ROTR32(x, 7) ^ ROTR32(x, 18) ^ SPH_T32((x) >> 3))
#define SSG2_1(x)      (ROTR32(x, 17) ^ ROTR32(x, 19) ^ SPH_T32((x) >> 10))

#define CH(X, Y, Z)    ((((Y) ^ (Z)) & (X)) ^ (Z))
#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))

#define SHA2_IN(x) hash.h4[x]


__global__ void sha256_gpu_hash_32(int threads, uint64_t startNounce, uint32_t *g_block, uint64_t *g_hash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint64_t nounce = startNounce + thread * 0x100000000ULL;

			
union {
uint8_t h1[32];
uint32_t h4[16];
uint64_t h8[8];
} hash;  

	
	uint32_t W[16]; 
        uint32_t r[8];	
        
#pragma unroll 16
	for (int i=0;i<16;i++) {
		hash.h4[i]= SWAP32(g_block[i]);
	}

#pragma unroll 8
	for (int i = 0; i < 8; i ++) {
		r[i] = H_256[i];}

	SHA2_ROUND_BODY(SHA2_IN, r);

#pragma unroll 16
	for (int i=0;i<16;i++) {
		hash.h4[i]= SWAP32(g_block[i+16]);
	}

	hash.h4[12] = SWAP32((nounce>>0));
	hash.h4[13] = SWAP32((nounce>>32));


	SHA2_ROUND_BODY(SHA2_IN, r);

		 
	for (int i=0;i<16;i++) {
		hash.h4[i]= SWAP32(g_block[i+32]);
	}

	//hash.h4[7] |= 0x80<<24;
	//hash.h4[15]= 122;
	SHA2_ROUND_BODY(SHA2_IN, r);
	






#if 0
	for(int i=0; i < 16; i++)
		printf("%8.8X", W[i]);
	printf("\n");

	for(int i=0; i < 8; i++)
		printf("%8.8X", r[i]);
	printf("\n");
#endif

#pragma unroll 8
	for(int i=0;i<8;i++) {	
		hash.h4[i] = SWAP32(r[i]);}

      
#pragma unroll 8
        for (int u = 0; u < 4; u ++) 
            g_hash[u*threads+thread] = hash.h8[u];    

        for (int u = 4; u < 8; u ++) 
            g_hash[u*threads+thread] = 0;    

 }
}

__global__ void sha256_gpu_full(int threads, uint64_t *g_data, uint64_t *g_hash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {

			
union {
uint8_t h1[64];
uint32_t h4[16];
uint64_t h8[8];
} hash;  

	uint32_t W[16]; 
        uint32_t r[8];	
#if 0	

#else
#pragma unroll 8
	for (int i = 0; i < 8; i ++) {
		r[i] = H_256[i];}

	uint32_t bytes = 0;
	for(int i=37; i >= 0; i--){
	    uint64_t data = g_data[i*threads+thread];
	    if(data==0)
		continue;
            if(data >> 56)
		bytes = (i+1)*8;
	    else if(data >> 48)
		bytes = i*8 + 7;
	    else if(data >> 40)
		bytes = i*8 + 6;
	    else if(data >> 32)
		bytes = i*8 + 5;
	    else if(data >> 24)
		bytes = i*8 + 4;
	    else if(data >> 16)
		bytes = i*8 + 3;
	    else if(data >> 8)
		bytes = i*8 + 2;
	    else
		bytes = i*8 + 1;
	    break;	
 	}

	uint32_t remain_bytes = bytes;
	uint32_t data_pos = 0;
	for(int j=0; j < 6; j++){
	    int i;
	    for(i=0; i < 8; i++){
		hash.h8[i] = 0;
	    }


	    for(i=0; i < 512;){
		if(remain_bytes){
		    uint32_t idx = data_pos/8;
		    hash.h8[i/64] = g_data[idx*threads+thread];
		    uint32_t consumed = remain_bytes > 8 ? 8 : remain_bytes;
		    data_pos += consumed; 
		    remain_bytes -= consumed;
		    i+=consumed*8;
		}else
	  	    break;
	    }

	    if(i < 512) { //Need padding bytes
		hash.h1[bytes % 64] = 0x80;
	    }

	    bool can_finish=false;
	    if(i < 448){
		//Will be done this cycle
		can_finish = true;
		hash.h4[15] = SWAP32(bytes*8);
	    }

#pragma unroll 16
	    for (int i=0;i<16;i++) {
		hash.h4[i]= SWAP32(hash.h4[i]);
	    }

  	    SHA2_ROUND_BODY(SHA2_IN, r);

	    if(can_finish)
		break;
	}

#pragma unroll 8
	for(int i=0;i<8;i++) {	
		hash.h4[i] = SWAP32(r[i]);}

        for (int u = 0; u < 4; u ++) 
            g_hash[u*threads+thread] = hash.h8[u];    

        for (int u = 4; u < 8; u ++) 
            g_hash[u*threads+thread] = 0; 

//	g_hash[thread] = bytes;
//        for (int u = 1; u < 8; u ++) 
//            g_hash[u*threads+thread] = 0;  

#endif
 }
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

void sha256_cpu_init(int thr_id, int threads, ctx* pctx)
{

    gpuErrchk(cudaMemcpyToSymbol(K_256,K256,sizeof(K256),0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(H_256,H256,sizeof(H256),0, cudaMemcpyHostToDevice));	

    gpuErrchk(cudaMalloc( (void**)&pctx->sha256_dblock, 192 )); 
	
}

__host__ void sha256_cpu_fullhash(int thr_id, int threads, uint64_t* data, uint64_t *d_hash)
{

	const int threadsperblock = 256; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
//	dim3 grid(1);
//	dim3 block(1);
//	size_t shared_size = 80*sizeof(uint64_t);
	size_t shared_size =0;
  	sha256_gpu_full<<<grid, block, shared_size>>>(threads, data, d_hash) ;


//	cudaStreamSynchronize(0);
//	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void sha256_cpu_hash_242(int thr_id, int threads, uint64_t startNounce, uint32_t* dblock, uint64_t *d_hash)
{

	const int threadsperblock = 256; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
//	dim3 grid(1);
//	dim3 block(1);
//	size_t shared_size = 80*sizeof(uint64_t);
	size_t shared_size =0;
  sha256_gpu_hash_32<<<grid, block, shared_size>>>(threads, startNounce, dblock, d_hash) ;


//	cudaStreamSynchronize(0);
//	MyStreamSynchronize(NULL, order, thr_id);
}

void sha256_scanhash(int throughput, uint64_t startNounce, CBlockHeader *hdr, uint64_t *d_hash, ctx* pctx){
	char block[192];
	uint64_t hash[8];

	memset(block,0,sizeof(block));
	memcpy(block,hdr,sizeof(*hdr));

	block[122] = 0x80;
	((uint32_t*)block)[192/4 - 1] = SWAP(976);

	gpuErrchk(cudaMemcpy( pctx->sha256_dblock, block, sizeof(block), cudaMemcpyHostToDevice )); 

	sha256_cpu_hash_242(0,throughput,startNounce,pctx->sha256_dblock,d_hash);
}

void sha256_fullhash(int throughput, uint64_t *data, uint64_t *hash){
	sha256_cpu_fullhash(0,throughput,data,hash);
}


