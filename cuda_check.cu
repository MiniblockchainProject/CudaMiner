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

__global__ void gpu_check(int threads, uint64_t *data, uint32_t *results, uint64_t target)
{
    __shared__ uint32_t tmp[512/32];

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);

    if(threadIdx.x < (512/32))
	tmp[threadIdx.x] = 0;

    __syncthreads();

    if (thread < threads)
    {
	uint64_t highword = data[threads*3 + thread];
	if(highword < target){
		atomicOr(&tmp[threadIdx.x/32], 1 << (threadIdx.x%32));
	}

	__syncthreads();
	if(threadIdx.x < (512/32))
		results[blockIdx.x*(4096/32) + threadIdx.x] = tmp[threadIdx.x];
    }
}

__host__ void checkhash(int threads, uint64_t *data, uint32_t *results, uint64_t target)
{

	const int threadsperblock = 512; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
//	dim3 grid(1);
//	dim3 block(1);
//	size_t shared_size = 80*sizeof(uint64_t);
	size_t shared_size =0;
  	gpu_check<<<grid, block, shared_size>>>(threads, data, results, target) ;


//	cudaStreamSynchronize(0);
//	MyStreamSynchronize(NULL, order, thr_id);
}

