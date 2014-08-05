#ifndef TRASHMINER_H
#define TRASHMINER_H

uint64_t swap_uint64( uint64_t val );

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
