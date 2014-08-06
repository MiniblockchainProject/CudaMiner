/*
 * sha512 djm34
 * 
 */

/*
 * sha-512 kernel implementation.
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
#include "sph_ripemd.h"

#include "trashminer.h"

#define USE_SHARED 1
#include "cuda_helper.h"

#define SPH_C32(x)    ((uint64_t)(x ## U))
//#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#define SPH_T32(x)    ((uint32_t)(x))

__device__ __forceinline__ uint64_t SWAP64(uint64_t x)
{
	// Input:	77665544 33221100
	// Output:	00112233 44556677
	uint64_t temp[2];
	temp[0] = __byte_perm(HIWORD(x), 0, 0x0123);
	temp[1] = __byte_perm(LOWORD(x), 0, 0x0123);

	return temp[0] | (temp[1]<<32);
}

#if 1
static __device__ __forceinline__ uint32_t SWAP32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}
#endif

#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))


// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);


/*
 * Round functions for RIPEMD-128 and RIPEMD-160.
 */
#define F1(x, y, z)   ((x) ^ (y) ^ (z))
#define F2(x, y, z)   ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   (((x) | ~(y)) ^ (z))
#define F4(x, y, z)   ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   ((x) ^ ((y) | ~(z)))

#define K11    SPH_C32(0x00000000)
#define K12    SPH_C32(0x5A827999)
#define K13    SPH_C32(0x6ED9EBA1)
#define K14    SPH_C32(0x8F1BBCDC)
#define K15    SPH_C32(0xA953FD4E)

#define K21    SPH_C32(0x50A28BE6)
#define K22    SPH_C32(0x5C4DD124)
#define K23    SPH_C32(0x6D703EF3)
#define K24    SPH_C32(0x7A6D76E9)
#define K25    SPH_C32(0x00000000)

#define RR(a, b, c, d, e, f, s, r, k)   do { \
		a = SPH_T32(ROTL(SPH_T32(a + f(b, c, d) + r + k), s) + e); \
		c = ROTL(c, 10); \
	} while (0)

#define ROUND1(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 1, b ## 1, c ## 1, d ## 1, e ## 1, f, s, r, K1 ## k)

#define ROUND2(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 2, b ## 2, c ## 2, d ## 2, e ## 2, f, s, r, K2 ## k)

/*
 * This macro defines the body for a RIPEMD-160 compression function
 * implementation. The "in" parameter should evaluate, when applied to a
 * numerical input parameter from 0 to 15, to an expression which yields
 * the corresponding input block. The "h" parameter should evaluate to
 * an array or pointer expression designating the array of 5 words which
 * contains the input and output of the compression function.
 */

#define RIPEMD160_ROUND_BODY(in, h)   do { \
		sph_u32 A1, B1, C1, D1, E1; \
		sph_u32 A2, B2, C2, D2, E2; \
		sph_u32 tmp; \
 \
		A1 = A2 = (h)[0]; \
		B1 = B2 = (h)[1]; \
		C1 = C2 = (h)[2]; \
		D1 = D2 = (h)[3]; \
		E1 = E2 = (h)[4]; \
 \
		ROUND1(A, B, C, D, E, F1, 11, in( 0),  1); \
		ROUND1(E, A, B, C, D, F1, 14, in( 1),  1); \
		ROUND1(D, E, A, B, C, F1, 15, in( 2),  1); \
		ROUND1(C, D, E, A, B, F1, 12, in( 3),  1); \
		ROUND1(B, C, D, E, A, F1,  5, in( 4),  1); \
		ROUND1(A, B, C, D, E, F1,  8, in( 5),  1); \
		ROUND1(E, A, B, C, D, F1,  7, in( 6),  1); \
		ROUND1(D, E, A, B, C, F1,  9, in( 7),  1); \
		ROUND1(C, D, E, A, B, F1, 11, in( 8),  1); \
		ROUND1(B, C, D, E, A, F1, 13, in( 9),  1); \
		ROUND1(A, B, C, D, E, F1, 14, in(10),  1); \
		ROUND1(E, A, B, C, D, F1, 15, in(11),  1); \
		ROUND1(D, E, A, B, C, F1,  6, in(12),  1); \
		ROUND1(C, D, E, A, B, F1,  7, in(13),  1); \
		ROUND1(B, C, D, E, A, F1,  9, in(14),  1); \
		ROUND1(A, B, C, D, E, F1,  8, in(15),  1); \
 \
		ROUND1(E, A, B, C, D, F2,  7, in( 7),  2); \
		ROUND1(D, E, A, B, C, F2,  6, in( 4),  2); \
		ROUND1(C, D, E, A, B, F2,  8, in(13),  2); \
		ROUND1(B, C, D, E, A, F2, 13, in( 1),  2); \
		ROUND1(A, B, C, D, E, F2, 11, in(10),  2); \
		ROUND1(E, A, B, C, D, F2,  9, in( 6),  2); \
		ROUND1(D, E, A, B, C, F2,  7, in(15),  2); \
		ROUND1(C, D, E, A, B, F2, 15, in( 3),  2); \
		ROUND1(B, C, D, E, A, F2,  7, in(12),  2); \
		ROUND1(A, B, C, D, E, F2, 12, in( 0),  2); \
		ROUND1(E, A, B, C, D, F2, 15, in( 9),  2); \
		ROUND1(D, E, A, B, C, F2,  9, in( 5),  2); \
		ROUND1(C, D, E, A, B, F2, 11, in( 2),  2); \
		ROUND1(B, C, D, E, A, F2,  7, in(14),  2); \
		ROUND1(A, B, C, D, E, F2, 13, in(11),  2); \
		ROUND1(E, A, B, C, D, F2, 12, in( 8),  2); \
 \
		ROUND1(D, E, A, B, C, F3, 11, in( 3),  3); \
		ROUND1(C, D, E, A, B, F3, 13, in(10),  3); \
		ROUND1(B, C, D, E, A, F3,  6, in(14),  3); \
		ROUND1(A, B, C, D, E, F3,  7, in( 4),  3); \
		ROUND1(E, A, B, C, D, F3, 14, in( 9),  3); \
		ROUND1(D, E, A, B, C, F3,  9, in(15),  3); \
		ROUND1(C, D, E, A, B, F3, 13, in( 8),  3); \
		ROUND1(B, C, D, E, A, F3, 15, in( 1),  3); \
		ROUND1(A, B, C, D, E, F3, 14, in( 2),  3); \
		ROUND1(E, A, B, C, D, F3,  8, in( 7),  3); \
		ROUND1(D, E, A, B, C, F3, 13, in( 0),  3); \
		ROUND1(C, D, E, A, B, F3,  6, in( 6),  3); \
		ROUND1(B, C, D, E, A, F3,  5, in(13),  3); \
		ROUND1(A, B, C, D, E, F3, 12, in(11),  3); \
		ROUND1(E, A, B, C, D, F3,  7, in( 5),  3); \
		ROUND1(D, E, A, B, C, F3,  5, in(12),  3); \
 \
		ROUND1(C, D, E, A, B, F4, 11, in( 1),  4); \
		ROUND1(B, C, D, E, A, F4, 12, in( 9),  4); \
		ROUND1(A, B, C, D, E, F4, 14, in(11),  4); \
		ROUND1(E, A, B, C, D, F4, 15, in(10),  4); \
		ROUND1(D, E, A, B, C, F4, 14, in( 0),  4); \
		ROUND1(C, D, E, A, B, F4, 15, in( 8),  4); \
		ROUND1(B, C, D, E, A, F4,  9, in(12),  4); \
		ROUND1(A, B, C, D, E, F4,  8, in( 4),  4); \
		ROUND1(E, A, B, C, D, F4,  9, in(13),  4); \
		ROUND1(D, E, A, B, C, F4, 14, in( 3),  4); \
		ROUND1(C, D, E, A, B, F4,  5, in( 7),  4); \
		ROUND1(B, C, D, E, A, F4,  6, in(15),  4); \
		ROUND1(A, B, C, D, E, F4,  8, in(14),  4); \
		ROUND1(E, A, B, C, D, F4,  6, in( 5),  4); \
		ROUND1(D, E, A, B, C, F4,  5, in( 6),  4); \
		ROUND1(C, D, E, A, B, F4, 12, in( 2),  4); \
 \
		ROUND1(B, C, D, E, A, F5,  9, in( 4),  5); \
		ROUND1(A, B, C, D, E, F5, 15, in( 0),  5); \
		ROUND1(E, A, B, C, D, F5,  5, in( 5),  5); \
		ROUND1(D, E, A, B, C, F5, 11, in( 9),  5); \
		ROUND1(C, D, E, A, B, F5,  6, in( 7),  5); \
		ROUND1(B, C, D, E, A, F5,  8, in(12),  5); \
		ROUND1(A, B, C, D, E, F5, 13, in( 2),  5); \
		ROUND1(E, A, B, C, D, F5, 12, in(10),  5); \
		ROUND1(D, E, A, B, C, F5,  5, in(14),  5); \
		ROUND1(C, D, E, A, B, F5, 12, in( 1),  5); \
		ROUND1(B, C, D, E, A, F5, 13, in( 3),  5); \
		ROUND1(A, B, C, D, E, F5, 14, in( 8),  5); \
		ROUND1(E, A, B, C, D, F5, 11, in(11),  5); \
		ROUND1(D, E, A, B, C, F5,  8, in( 6),  5); \
		ROUND1(C, D, E, A, B, F5,  5, in(15),  5); \
		ROUND1(B, C, D, E, A, F5,  6, in(13),  5); \
 \
		ROUND2(A, B, C, D, E, F5,  8, in( 5),  1); \
		ROUND2(E, A, B, C, D, F5,  9, in(14),  1); \
		ROUND2(D, E, A, B, C, F5,  9, in( 7),  1); \
		ROUND2(C, D, E, A, B, F5, 11, in( 0),  1); \
		ROUND2(B, C, D, E, A, F5, 13, in( 9),  1); \
		ROUND2(A, B, C, D, E, F5, 15, in( 2),  1); \
		ROUND2(E, A, B, C, D, F5, 15, in(11),  1); \
		ROUND2(D, E, A, B, C, F5,  5, in( 4),  1); \
		ROUND2(C, D, E, A, B, F5,  7, in(13),  1); \
		ROUND2(B, C, D, E, A, F5,  7, in( 6),  1); \
		ROUND2(A, B, C, D, E, F5,  8, in(15),  1); \
		ROUND2(E, A, B, C, D, F5, 11, in( 8),  1); \
		ROUND2(D, E, A, B, C, F5, 14, in( 1),  1); \
		ROUND2(C, D, E, A, B, F5, 14, in(10),  1); \
		ROUND2(B, C, D, E, A, F5, 12, in( 3),  1); \
		ROUND2(A, B, C, D, E, F5,  6, in(12),  1); \
 \
		ROUND2(E, A, B, C, D, F4,  9, in( 6),  2); \
		ROUND2(D, E, A, B, C, F4, 13, in(11),  2); \
		ROUND2(C, D, E, A, B, F4, 15, in( 3),  2); \
		ROUND2(B, C, D, E, A, F4,  7, in( 7),  2); \
		ROUND2(A, B, C, D, E, F4, 12, in( 0),  2); \
		ROUND2(E, A, B, C, D, F4,  8, in(13),  2); \
		ROUND2(D, E, A, B, C, F4,  9, in( 5),  2); \
		ROUND2(C, D, E, A, B, F4, 11, in(10),  2); \
		ROUND2(B, C, D, E, A, F4,  7, in(14),  2); \
		ROUND2(A, B, C, D, E, F4,  7, in(15),  2); \
		ROUND2(E, A, B, C, D, F4, 12, in( 8),  2); \
		ROUND2(D, E, A, B, C, F4,  7, in(12),  2); \
		ROUND2(C, D, E, A, B, F4,  6, in( 4),  2); \
		ROUND2(B, C, D, E, A, F4, 15, in( 9),  2); \
		ROUND2(A, B, C, D, E, F4, 13, in( 1),  2); \
		ROUND2(E, A, B, C, D, F4, 11, in( 2),  2); \
 \
		ROUND2(D, E, A, B, C, F3,  9, in(15),  3); \
		ROUND2(C, D, E, A, B, F3,  7, in( 5),  3); \
		ROUND2(B, C, D, E, A, F3, 15, in( 1),  3); \
		ROUND2(A, B, C, D, E, F3, 11, in( 3),  3); \
		ROUND2(E, A, B, C, D, F3,  8, in( 7),  3); \
		ROUND2(D, E, A, B, C, F3,  6, in(14),  3); \
		ROUND2(C, D, E, A, B, F3,  6, in( 6),  3); \
		ROUND2(B, C, D, E, A, F3, 14, in( 9),  3); \
		ROUND2(A, B, C, D, E, F3, 12, in(11),  3); \
		ROUND2(E, A, B, C, D, F3, 13, in( 8),  3); \
		ROUND2(D, E, A, B, C, F3,  5, in(12),  3); \
		ROUND2(C, D, E, A, B, F3, 14, in( 2),  3); \
		ROUND2(B, C, D, E, A, F3, 13, in(10),  3); \
		ROUND2(A, B, C, D, E, F3, 13, in( 0),  3); \
		ROUND2(E, A, B, C, D, F3,  7, in( 4),  3); \
		ROUND2(D, E, A, B, C, F3,  5, in(13),  3); \
 \
		ROUND2(C, D, E, A, B, F2, 15, in( 8),  4); \
		ROUND2(B, C, D, E, A, F2,  5, in( 6),  4); \
		ROUND2(A, B, C, D, E, F2,  8, in( 4),  4); \
		ROUND2(E, A, B, C, D, F2, 11, in( 1),  4); \
		ROUND2(D, E, A, B, C, F2, 14, in( 3),  4); \
		ROUND2(C, D, E, A, B, F2, 14, in(11),  4); \
		ROUND2(B, C, D, E, A, F2,  6, in(15),  4); \
		ROUND2(A, B, C, D, E, F2, 14, in( 0),  4); \
		ROUND2(E, A, B, C, D, F2,  6, in( 5),  4); \
		ROUND2(D, E, A, B, C, F2,  9, in(12),  4); \
		ROUND2(C, D, E, A, B, F2, 12, in( 2),  4); \
		ROUND2(B, C, D, E, A, F2,  9, in(13),  4); \
		ROUND2(A, B, C, D, E, F2, 12, in( 9),  4); \
		ROUND2(E, A, B, C, D, F2,  5, in( 7),  4); \
		ROUND2(D, E, A, B, C, F2, 15, in(10),  4); \
		ROUND2(C, D, E, A, B, F2,  8, in(14),  4); \
 \
		ROUND2(B, C, D, E, A, F1,  8, in(12),  5); \
		ROUND2(A, B, C, D, E, F1,  5, in(15),  5); \
		ROUND2(E, A, B, C, D, F1, 12, in(10),  5); \
		ROUND2(D, E, A, B, C, F1,  9, in( 4),  5); \
		ROUND2(C, D, E, A, B, F1, 12, in( 1),  5); \
		ROUND2(B, C, D, E, A, F1,  5, in( 5),  5); \
		ROUND2(A, B, C, D, E, F1, 14, in( 8),  5); \
		ROUND2(E, A, B, C, D, F1,  6, in( 7),  5); \
		ROUND2(D, E, A, B, C, F1,  8, in( 6),  5); \
		ROUND2(C, D, E, A, B, F1, 13, in( 2),  5); \
		ROUND2(B, C, D, E, A, F1,  6, in(13),  5); \
		ROUND2(A, B, C, D, E, F1,  5, in(14),  5); \
		ROUND2(E, A, B, C, D, F1, 15, in( 0),  5); \
		ROUND2(D, E, A, B, C, F1, 13, in( 3),  5); \
		ROUND2(C, D, E, A, B, F1, 11, in( 9),  5); \
		ROUND2(B, C, D, E, A, F1, 11, in(11),  5); \
 \
		tmp = SPH_T32((h)[1] + C1 + D2); \
		(h)[1] = SPH_T32((h)[2] + D1 + E2); \
		(h)[2] = SPH_T32((h)[3] + E1 + A2); \
		(h)[3] = SPH_T32((h)[4] + A1 + B2); \
		(h)[4] = SPH_T32((h)[0] + B1 + C2); \
		(h)[0] = tmp; \
	} while (0)

__global__ void ripemd_gpu_hash_242(int threads, uint64_t startNounce, uint32_t *g_block, uint64_t *g_hash)
{
/*
     __shared__ uint64_t sharedMem[80];
	
		sharedMem[threadIdx.x]      = K_512[threadIdx.x];
*/	

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t *inpHash = g_block;
		
			
union {
uint8_t h1[32];
uint32_t h4[8];
uint64_t h8[4];
} hash;  

		
        
		 
	
		
	uint32_t W[16]; 
        uint32_t r[5];

#pragma unroll 16
	uint64_t nonce = startNounce + thread * 0x100000000ULL;

	for (int i = 0; i < 16; i ++) {
		W[i] = (inpHash[i]);
	}


	r[0] = SPH_C32(0x67452301);
	r[1] = SPH_C32(0xEFCDAB89);
	r[2] = SPH_C32(0x98BADCFE);
	r[3] = SPH_C32(0x10325476);
	r[4] = SPH_C32(0xC3D2E1F0);

#define IN(x) W[x]

	RIPEMD160_ROUND_BODY(IN,r);

#pragma unroll 16
	for (int i = 0; i < 16; i ++) {
		W[i] = (inpHash[i+16]);
	}

	W[12] = nonce;
	W[13] = nonce >> 32;

	RIPEMD160_ROUND_BODY(IN,r);

#pragma unroll 16
	for (int i = 0; i < 16; i ++) {
		W[i] = (inpHash[i+32]);
	}

	RIPEMD160_ROUND_BODY(IN,r);


#pragma unroll 5
	for(int i=0;i<5;i++) {	
		hash.h4[i] = (r[i]);}
	hash.h4[5] = 0;

      
#pragma unroll 3
      for (int u = 0; u < 3; u ++) 
            g_hash[u*threads+thread] = hash.h8[u];    

#pragma unroll 5
      for (int u = 3; u < 8; u ++) 
            g_hash[u*threads+thread] = 0;  
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


void ripemd_cpu_init(int thr_id, int threads, ctx* pctx)
{
	gpuErrchk(cudaMalloc( (void**)&pctx->ripemd_dblock,192 )); 

}


__host__ void ripemd_cpu_hash_242(int thr_id, int threads, uint64_t startNounce, uint32_t *d_block, uint64_t *d_hash)
{

	const int threadsperblock = 256; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
//	dim3 grid(1);
//	dim3 block(1);
//	size_t shared_size = 80*sizeof(uint64_t);
	size_t shared_size =0;
	ripemd_gpu_hash_242<<<grid, block, shared_size>>>(threads, startNounce, d_block, d_hash);

//	cudaStreamSynchronize(0);
	MyStreamSynchronize(NULL, 7, thr_id);
}

void ripemd_scanhash(int throughput, uint64_t startNonce, CBlockHeader *hdr, uint64_t *d_hash, ctx* pctx){
	char block[192];
	uint64_t hash[8];

	memset(block,0,sizeof(block));
	memcpy(block,hdr,sizeof(*hdr));

	block[122] = 0x80;
	((uint32_t*)block)[192/4 - 2] = 976;

	gpuErrchk(cudaMemcpyAsync( pctx->ripemd_dblock, block, sizeof(block), cudaMemcpyHostToDevice, 0 )); 

	ripemd_cpu_hash_242(pctx->thr_id,throughput,startNonce,pctx->ripemd_dblock,d_hash);

}

