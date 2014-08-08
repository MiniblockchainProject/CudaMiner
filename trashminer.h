#ifndef TRASHMINER_H
#define TRASHMINER_H

uint64_t swap_uint64( uint64_t val );

cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

//Storage for result of sha256
struct ctx {
	uint64_t *d_hash[8];
	uint64_t *d_prod[2];
	uint64_t *hash[8];
	uint64_t *prod[2];
	cudaStream_t stream;

	uint32_t* sha256_dblock;
	uint32_t* sha512_dblock;
	uint32_t* keccak_dblock;
	uint32_t* whirlpool_dblock;
	uint32_t* haval_dblock;
	uint32_t* tiger_dblock;
	uint32_t* ripemd_dblock;

	int thr_id;

	uint64_t* keccak_dstate;
	uint32_t* results;
	uint32_t* d_results;
};

#pragma pack(push,1)
class CBlockHeader
{
public:
    //!!!!!!!!!!! struct must be in packed order even though serialize order is version first
    //or else we can't use hash macros, could also use #pragma pack but that has 
    //terrible implicatation on non-x86
    uint256 hashPrevBlock; //32
    uint256 hashMerkleRoot; //64
    uint256 hashAccountRoot; //96
    uint64_t nTime; //104
    uint64_t nHeight; //112
    uint64_t nNonce; //120
    uint16_t nVersion; //122
};
#pragma pack(pop)



#endif //TRASHMINER_H
