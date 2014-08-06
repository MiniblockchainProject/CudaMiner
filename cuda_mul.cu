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

#define USE_SHARED 1
#include "cuda_helper.h"

#include "trashminer.h"

#define umul_ppmm(h,l,m,n) \
	h = __umul64hi(m,n); \
	l = m * n;


__device__ void mulScalar(uint32_t thread, uint32_t threads, uint32_t len, uint64_t* g_p, uint64_t* g_v, uint64_t sml, uint32_t *size){
  uint64_t ul, cl, hpl, lpl;
  uint32_t i;
  cl = 0;
  for(i=0; i < len; i++) {
      ul = g_v[i*threads + thread];
      umul_ppmm (hpl, lpl, ul, sml);

      lpl += cl;
      cl = (lpl < cl) + hpl;

      g_p[i*threads + thread] = lpl;
    }

    g_p[len*threads + thread] = cl;
    *size = len + (cl != 0);
}

uint64_t __device__ addmul_1g (uint32_t thread, uint32_t threads, uint64_t *sum, uint32_t sofst, uint64_t *x, uint64_t xsz, uint64_t a){
	uint64_t carry=0;
	uint32_t i;
	uint64_t ul,lpl,hpl,rl;

	for(i=0; i < xsz; i++){
		
      		ul = x[i*threads + thread];
      		umul_ppmm (hpl, lpl, ul, a);

      		lpl += carry;
      		carry = (lpl < carry) + hpl;

      		rl = sum[(i+sofst) * threads + thread];
      		lpl = rl + lpl;
      		carry += lpl < rl;
      		sum[(i+sofst)*threads + thread] = lpl;
    	}

  	return carry;
}


__global__ void gpu_mul(int threads, uint32_t ulegs, uint32_t vlegs, uint64_t *g_u, uint64_t *g_v, uint64_t *g_p)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
	if(ulegs < vlegs){
		uint64_t t1=ulegs;
		ulegs = vlegs;
		vlegs = t1;

		uint64_t *t2 = g_u;
		g_u = g_v;
		g_v = t2;
	}

	uint32_t vofst=1,rofst=1,psize=0;
	mulScalar(thread,threads,ulegs,g_p,g_u,g_v[thread],&psize);   

#if 1

  	while (vofst < vlegs) {
		//clear high word //TODO: right 
	//	printf("Size: %d\n", rp->size[tid]);
	    	g_p[(psize+0)*threads+thread] = 0;

            	g_p[(ulegs+rofst)*threads + thread] = addmul_1g (thread, threads, g_p ,rofst , g_u, ulegs, g_v[vofst*threads+thread]);
	    	vofst++; rofst++;
	    	psize++;
        }

//	if(D_REF(rp->d,up->size[tid] + vp->size[tid] - 1,tid) != (uint64_t)0)
//		rp->size[tid]++;


#endif
    }
}


__host__ void cpu_mul(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p)
{

	const int threadsperblock = 256; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
//	dim3 grid(1);
//	dim3 block(1);
//	size_t shared_size = 80*sizeof(uint64_t);
	size_t shared_size =0;
  	gpu_mul<<<grid, block, shared_size>>>(threads, alegs, blegs, g_a, g_b, g_p) ;


//	cudaStreamSynchronize(0);
//	MyStreamSynchronize(NULL, order, thr_id);
}

