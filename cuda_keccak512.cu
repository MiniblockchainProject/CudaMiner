#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "uint256.h"
#include "sph_keccak.h"

#include "trashminer.h"

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

#include "cuda_helper.h"

#define U32TO64_LE(p) \
     (  ((uint64_t)(*p)) | (((uint64_t)(*(p + 1)) << 32)))

#define U64TO32_LE(p, v) \
    *p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

__device__ __forceinline__ uint64_t SWAP64(uint64_t x)
{
	// Input:	77665544 33221100
	// Output:	00112233 44556677
	uint64_t temp[2];
	temp[0] = __byte_perm(HIWORD(x), 0, 0x0123);
	temp[1] = __byte_perm(LOWORD(x), 0, 0x0123);

	return temp[0] | (temp[1]<<32);
}

static const uint64_t host_keccak_round_constants[24] = {
    0x0000000000000001ull, 0x0000000000008082ull,
    0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull,
    0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull,
    0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull,
    0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull,
    0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull,
    0x0000000080000001ull, 0x8000000080008008ull
};

__constant__ uint64_t c_keccak_round_constants[24];

static __device__  void
keccak_block(uint64_t *s, const uint64_t *keccak_round_constants) {
    int i;
    uint64_t t[5], u[5], v, w;

    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        u[0] = t[4] ^ ROTL64(t[1], 1);
        u[1] = t[0] ^ ROTL64(t[2], 1);
        u[2] = t[1] ^ ROTL64(t[3], 1);
        u[3] = t[2] ^ ROTL64(t[4], 1);
        u[4] = t[3] ^ ROTL64(t[0], 1);

        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
        v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
        v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
        v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }
}

__global__ void keccak512_gpu_hash_242(int threads, uint64_t startNounce, uint64_t *g_state, uint32_t *g_block, uint64_t *g_hash)
{
        // State initialisieren
__align__(16)    uint64_t keccak_gpu_state[25];

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t *inpHash = (uint32_t*)g_block;

#pragma unroll 25
        for (int i=0; i<25; i++)
            keccak_gpu_state[i] = g_state[i];

//Round 2
    	/* absorb input */
#pragma unroll 9
	uint64_t nonce = startNounce+thread * 0x100000000ULL;
    	for (int i = 0; i < 72 / 8; i++){
		if(i==5)
			keccak_gpu_state[i] ^= nonce;
		else
	        	keccak_gpu_state[i] ^= U32TO64_LE(&inpHash[i*2+18]);
    	}


        // den Block einmal gut durchschütteln
        keccak_block(keccak_gpu_state, c_keccak_round_constants);


#pragma unroll 8
        for (size_t i = 0; i < 8; i ++) {
    //        keccak_gpu_state[i] = SWAP64(keccak_gpu_state[i]);
        }

#pragma unroll 8
        for(int i=0;i<8;i++)
            g_hash[i*threads + thread] = keccak_gpu_state[i];
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


// Setup-Funktionen
__host__ void keccak512_cpu_init(int thr_id, int threads, ctx* pctx)
{
    // Kopiere die Hash-Tabellen in den GPU-Speicher
    cudaMemcpyToSymbol( c_keccak_round_constants,
                        host_keccak_round_constants,
                        sizeof(host_keccak_round_constants),
                        0, cudaMemcpyHostToDevice);

    gpuErrchk(cudaMalloc( (void**)&pctx->keccak_dblock, 144 )); 
    gpuErrchk(cudaMalloc( (void**)&pctx->keccak_dstate, 25*8 )); 
}

__host__ void keccak512_cpu_hash_242(int thr_id, int threads, uint64_t startNounce, uint64_t* dstate, uint32_t *d_block, uint64_t *d_hash)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    keccak512_gpu_hash_242<<<grid, block, shared_size>>>(threads, startNounce, dstate, d_block, (uint64_t*)d_hash);

 //   cudaStreamSynchronize(0);
    MyStreamSynchronize(NULL, 3, thr_id);
}

extern "C" {
void keccak_one(const void *data, uint64_t state[]);
}

void keccak512_scanhash(int throughput, uint64_t startNounce, CBlockHeader *hdr, uint64_t *d_hash, ctx* pctx){
	char block[144];
	uint64_t state[25];

	memset(block,0,sizeof(block));
	memcpy(block,hdr,sizeof(*hdr));

	block[122] = 0x1;
	((uint64_t*)block)[144/8 - 1] = 0x8000000000000000ULL;

	keccak_one(block,state);

#if 0
	printf("KeccakBlock: ");
	for(int i=0; i < (144+7)/8; i++){
		printf("%16.16llX", ((uint64_t*)block)[i]);
	}
	printf("\n");

	printf("Keccak: ");
	for(int i=0; i < 25; i++){
		printf("%16.16llX", state[i]);
	}
	printf("\n");
#endif
	gpuErrchk(cudaMemcpyAsync( pctx->keccak_dblock, block, sizeof(block), cudaMemcpyHostToDevice, 0 )); 
	gpuErrchk(cudaMemcpyAsync( pctx->keccak_dstate, state, sizeof(state), cudaMemcpyHostToDevice, 0 )); 

	keccak512_cpu_hash_242(pctx->thr_id,throughput,startNounce,pctx->keccak_dstate, pctx->keccak_dblock,d_hash);
}

