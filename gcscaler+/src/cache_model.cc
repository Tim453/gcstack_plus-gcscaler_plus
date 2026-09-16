#include "cache_model.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

using namespace GCoM;
using namespace std;
namespace fs = std::filesystem;

// For Accel-sim cache Mode
CacheModel::CacheModel(ECacheModelType type, fs::path cacheStatPath, std::streamoff cacheStatPos, unsigned kernelIdx, std::streamoff &nextKernelCacheStatPos)
{
    mType = type;
    assert(type == ECacheModelType::ACCELSIM);

    ifstream ifs(cacheStatPath, ios::binary);
    assert(ifs.is_open() == true);

    // find the cache statistic of kernelIdx in the binary file
    ifs.seekg(0, ios::end);
    streamoff fileSize = ifs.tellg();
    assert(cacheStatPos < fileSize);
    ifs.seekg(cacheStatPos, ios::beg);

    struct CacheStatHeader header;
    boost::archive::binary_iarchive ia(ifs);
    ia & header;
    assert(header.kernelNumber == (int) kernelIdx);
    ia & mKernelCacheStat;

    nextKernelCacheStatPos = ifs.tellg();

    // Detect v2 format and build PC-based lookup index
    if (!mKernelCacheStat.empty() && !mKernelCacheStat[0].empty()
        && mKernelCacheStat[0][0].serializationVersion >= 2)
    {
        mUseV2Lookup = true;
        mWarpPCIndex.resize(mKernelCacheStat.size());
        for (unsigned w = 0; w < mKernelCacheStat.size(); w++)
        {
            for (unsigned i = 0; i < mKernelCacheStat[w].size(); i++)
            {
                mWarpPCIndex[w][mKernelCacheStat[w][i].pc].push(i);
            }
        }
    }
}

// For Accel-sim cache Mode
// read cache statistics of Accel-sim
int CacheModel::cacheAccess(WarpInst &inst, unsigned  warpIdx, unsigned  instIdx)
{
    if (inst.mDecoded.op == EUArchOp::LOAD_OP || inst.mDecoded.op == EUArchOp::STORE_OP)
    {
        WarpInstCacheStat *matchedPtr = nullptr;

        if (mUseV2Lookup)
        {
            // v2: PC-based FIFO lookup — robust against instruction classification differences
            int pc = (int) inst.mDecoded.pc;
            auto &pcMap = mWarpPCIndex[warpIdx];
            auto it = pcMap.find(pc);
            if (it != pcMap.end() && !it->second.empty())
            {
                unsigned idx = it->second.front();
                it->second.pop();
                matchedPtr = &mKernelCacheStat[warpIdx][idx];
            }
        }
        else
        {
            unsigned &cursor = mWarpMemInstIdx[warpIdx];
            vector<WarpInstCacheStat> &warpCacheStat = mKernelCacheStat[warpIdx];

            if (cursor < warpCacheStat.size()
                    && warpCacheStat[cursor].warpInstIdx == (unsigned) -1)
            {
                matchedPtr = &warpCacheStat[cursor++];
            }
            else
            {
                while (cursor < warpCacheStat.size()
                        && warpCacheStat[cursor].warpInstIdx < instIdx)
                    cursor++;
                if (cursor < warpCacheStat.size()
                        && warpCacheStat[cursor].warpInstIdx == instIdx)
                    matchedPtr = &warpCacheStat[cursor++];
            }
        }

        if (matchedPtr == nullptr)
        {
            cout << "[Error] Accel-sim Cache model: no matching entry found"
                 << " warpIdx=" << warpIdx
                 << " | inst: pc=0x" << hex << inst.mDecoded.pc
                 << " op=" << dec << static_cast<int>(inst.mDecoded.op)
                 << endl;
            return 0;
        }

        if (inst.mDecoded.pc != (GCoM::Address) (*matchedPtr).pc)
        {
            cout << "[Error] Accel-sim Cache model: instruction mismatch"
                 << " warpIdx=" << warpIdx
                 << " | inst: pc=0x" << hex << inst.mDecoded.pc
                 << " op=" << dec << static_cast<int>(inst.mDecoded.op)
                 << " | matched: pc=0x" << hex << (*matchedPtr).pc
                 << " op=" << dec << static_cast<int>((*matchedPtr).op)
                 << endl;
        }

        inst.mMemStat = (*matchedPtr);
        for (Address addr : (*matchedPtr).accessQAddr)
        {
            MemAccess psuedoMemAccess;
            psuedoMemAccess.mAddr = addr;
            inst.mDecoded.accessQ.push_back(psuedoMemAccess);
        }

        // Collect warp instruction cache statistics
        if (inst.mDecoded.space == EMemorySpace::GLOBAL_SPACE || inst.mDecoded.space == EMemorySpace::LOCAL_SPACE)
        {
            auto key = make_pair(inst.mDecoded.pc, instIdx);
            // make element in mWarpInstStatistics_PCInstIdx2CacheHitMiss if not exist, else reference it
            Warp::WarpInstCacheHitMissCount &warpInstCStat = mWarpInstStatistics_PCInstIdx2CacheHitMiss[key];

            if (inst.mMemStat.l1Miss > 0)
                warpInstCStat.l1MissWarp += 1;
            else
                warpInstCStat.l1HitWarp += 1;

            if (inst.mMemStat.l2Miss > 0)
                warpInstCStat.l2MissWarp += 1;
            else
                warpInstCStat.l2HitWarp += 1;
        }
    }

    return 0;
}

int CacheModel::updateGlobalCacheStat(Warp &warp)
{
    for (unsigned instIdx = 0; instIdx < warp.mInsts.size(); instIdx++)
    {
        WarpInst &warpInst = warp.mInsts[instIdx];

        // find corresponding global cache statistics of a warp instruction
        // if not found, set to 0
        auto key = make_pair(warpInst.mDecoded.pc, instIdx);
        auto it = mWarpInstStatistics_PCInstIdx2CacheHitMiss.find(key);
        if (it != mWarpInstStatistics_PCInstIdx2CacheHitMiss.end())
            warp.mGlobalCacheStat.push_back(it->second);
        else
            warp.mGlobalCacheStat.push_back(Warp::WarpInstCacheHitMissCount());
    }
    return 0;
}

// Copide from Accel-sim hashing.cc
unsigned HashAdressWithIpolyFunction(Address higherBits, unsigned index, unsigned nBins)
{
    if (nBins == 16)
    {
        std::bitset<64> a(higherBits);
        std::bitset<4> b(index);
        std::bitset<4> newIndex(index);

        newIndex[0] =
            a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
        newIndex[1] =
            a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
        newIndex[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
        newIndex[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];

        return newIndex.to_ulong();
    }
    else if (nBins == 32)
    {
        std::bitset<64> a(higherBits);
        std::bitset<5> b(index);
        std::bitset<5> newIndex(index);

        newIndex[0] =
            a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
        newIndex[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^
                       a[1] ^ b[1];
        newIndex[2] =
            a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
        newIndex[3] =
            a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
        newIndex[4] =
            a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
        return newIndex.to_ulong();
    }
    else if (nBins == 64)
    {
        std::bitset<64> a(higherBits);
        std::bitset<6> b(index);
        std::bitset<6> newIndex(index);

        newIndex[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^
                       a[0] ^ b[0];
        newIndex[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^
                       a[0] ^ b[1];
        newIndex[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^
                       a[1] ^ b[2];
        newIndex[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^
                       a[2] ^ b[3];
        newIndex[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^
                       a[3] ^ b[4];
        newIndex[5] =
            a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
        return newIndex.to_ulong();
    }
else if (nBins == 128) {
    std::bitset<64> a(higherBits);
    std::bitset<7> b(index);
    std::bitset<7> new_index(index);

    // Equations derived for primitive polynomial x^7 + x^3 + 1 (IPOLY(131))
    new_index[0] = a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] = a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[1] ^ b[1];
    new_index[2] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[2] ^ b[2];
    new_index[3] = a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[3] ^ a[0] ^ b[3];
    new_index[4] = a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[4] ^ a[1] ^ b[4];
    new_index[5] = a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[5] ^ a[2] ^ b[5];
    new_index[6] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[6] ^ a[3] ^ b[6];

    return new_index.to_ulong();
  } else if (nBins == 256) {
    std::bitset<64> a(higherBits);
    std::bitset<8> b(index);
    std::bitset<8> new_index(index);

    // Equations derived for primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (IPOLY(285))
    new_index[0] = a[19] ^ a[18] ^ a[17] ^ a[15] ^ a[13] ^ a[12] ^ a[10] ^ a[7] ^ a[4] ^ a[3] ^ a[2] ^ a[0] ^ b[0];
    new_index[1] = a[20] ^ a[19] ^ a[18] ^ a[16] ^ a[14] ^ a[13] ^ a[11] ^ a[8] ^ a[5] ^ a[4] ^ a[3] ^ a[1] ^ b[1];
    new_index[2] = a[21] ^ a[20] ^ a[19] ^ a[17] ^ a[15] ^ a[14] ^ a[12] ^ a[9] ^ a[6] ^ a[5] ^ a[4] ^ a[2] ^ b[2];
    new_index[3] = a[22] ^ a[21] ^ a[20] ^ a[18] ^ a[16] ^ a[15] ^ a[13] ^ a[10] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ b[3];
    new_index[4] = a[23] ^ a[22] ^ a[21] ^ a[19] ^ a[17] ^ a[16] ^ a[14] ^ a[11] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[0] ^ b[4];
    new_index[5] = a[24] ^ a[23] ^ a[22] ^ a[20] ^ a[18] ^ a[17] ^ a[15] ^ a[12] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[1] ^ b[5];
    new_index[6] = a[25] ^ a[24] ^ a[23] ^ a[21] ^ a[19] ^ a[18] ^ a[16] ^ a[13] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[2] ^ b[6];
    new_index[7] = a[26] ^ a[25] ^ a[24] ^ a[22] ^ a[20] ^ a[19] ^ a[17] ^ a[14] ^ a[11] ^ a[10] ^ a[9] ^ a[7] ^ a[3] ^ b[7];

    return new_index.to_ulong();

  } else if (nBins == 512) {
    std::bitset<64> a(higherBits);
    std::bitset<9> b(index);
    std::bitset<9> new_index(index);

    // Equations derived for primitive polynomial x^9 + x^4 + 1 (IPOLY(529))
    new_index[0] = a[25] ^ a[23] ^ a[22] ^ a[21] ^ a[19] ^ a[17] ^ a[15] ^ a[12] ^ a[8] ^ a[4] ^ a[0] ^ b[0];
    new_index[1] = a[26] ^ a[24] ^ a[23] ^ a[22] ^ a[20] ^ a[18] ^ a[16] ^ a[13] ^ a[9] ^ a[5] ^ a[1] ^ b[1];
    new_index[2] = a[27] ^ a[25] ^ a[24] ^ a[23] ^ a[21] ^ a[19] ^ a[17] ^ a[14] ^ a[10] ^ a[6] ^ a[2] ^ b[2];
    new_index[3] = a[28] ^ a[26] ^ a[25] ^ a[24] ^ a[22] ^ a[20] ^ a[18] ^ a[15] ^ a[11] ^ a[7] ^ a[3] ^ b[3];
    new_index[4] = a[29] ^ a[27] ^ a[26] ^ a[25] ^ a[23] ^ a[21] ^ a[19] ^ a[16] ^ a[12] ^ a[8] ^ a[4] ^ a[0] ^ b[4];
    new_index[5] = a[30] ^ a[28] ^ a[27] ^ a[26] ^ a[24] ^ a[22] ^ a[20] ^ a[17] ^ a[13] ^ a[9] ^ a[5] ^ a[1] ^ b[5];
    new_index[6] = a[31] ^ a[29] ^ a[28] ^ a[27] ^ a[25] ^ a[23] ^ a[21] ^ a[18] ^ a[14] ^ a[10] ^ a[6] ^ a[2] ^ b[6];
    new_index[7] = a[32] ^ a[30] ^ a[29] ^ a[28] ^ a[26] ^ a[24] ^ a[22] ^ a[19] ^ a[15] ^ a[11] ^ a[7] ^ a[3] ^ b[7];
    new_index[8] = a[33] ^ a[31] ^ a[30] ^ a[29] ^ a[27] ^ a[25] ^ a[23] ^ a[20] ^ a[16] ^ a[12] ^ a[8] ^ a[4] ^ b[8];

    return new_index.to_ulong();

  } else if (nBins == 1024) {
    std::bitset<64> a(higherBits);
    std::bitset<10> b(index);
    std::bitset<10> new_index(index);

    // Equations derived for primitive polynomial x^10 + x^3 + 1 (IPOLY(1033))
    new_index[0] = a[30] ^ a[28] ^ a[27] ^ a[26] ^ a[24] ^ a[22] ^ a[20] ^ a[18] ^ a[16] ^ a[12] ^ a[9] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] = a[31] ^ a[29] ^ a[28] ^ a[27] ^ a[25] ^ a[23] ^ a[21] ^ a[19] ^ a[17] ^ a[13] ^ a[10] ^ a[4] ^ a[1] ^ b[1];
    new_index[2] = a[32] ^ a[30] ^ a[29] ^ a[28] ^ a[26] ^ a[24] ^ a[22] ^ a[20] ^ a[18] ^ a[14] ^ a[11] ^ a[5] ^ a[2] ^ b[2];
    new_index[3] = a[33] ^ a[31] ^ a[30] ^ a[29] ^ a[27] ^ a[25] ^ a[23] ^ a[21] ^ a[19] ^ a[15] ^ a[12] ^ a[6] ^ a[3] ^ a[0] ^ b[3];
    new_index[4] = a[34] ^ a[32] ^ a[31] ^ a[30] ^ a[28] ^ a[26] ^ a[24] ^ a[22] ^ a[20] ^ a[16] ^ a[13] ^ a[7] ^ a[4] ^ a[1] ^ b[4];
    new_index[5] = a[35] ^ a[33] ^ a[32] ^ a[31] ^ a[29] ^ a[27] ^ a[25] ^ a[23] ^ a[21] ^ a[17] ^ a[14] ^ a[8] ^ a[5] ^ a[2] ^ b[5];
    new_index[6] = a[36] ^ a[34] ^ a[33] ^ a[32] ^ a[30] ^ a[28] ^ a[26] ^ a[24] ^ a[22] ^ a[18] ^ a[15] ^ a[9] ^ a[6] ^ a[3] ^ b[6];
    new_index[7] = a[37] ^ a[35] ^ a[34] ^ a[33] ^ a[31] ^ a[29] ^ a[27] ^ a[25] ^ a[23] ^ a[19] ^ a[16] ^ a[10] ^ a[7] ^ a[4] ^ b[7];
    new_index[8] = a[38] ^ a[36] ^ a[35] ^ a[34] ^ a[32] ^ a[30] ^ a[28] ^ a[26] ^ a[24] ^ a[20] ^ a[17] ^ a[11] ^ a[8] ^ a[5] ^ b[8];
    new_index[9] = a[39] ^ a[37] ^ a[36] ^ a[35] ^ a[33] ^ a[31] ^ a[29] ^ a[27] ^ a[25] ^ a[21] ^ a[18] ^ a[12] ^ a[9] ^ a[6] ^ b[9];

    return new_index.to_ulong();
  } else { /* Else incorrect number of channels for the hashing function */
    assert(
        "\nmemory_partition_indexing error: The number of "
        "channels should be "
        "16, 32, 64, 128, 256, 512 or 1024 for the hashing IPOLY index function. other banks "
        "numbers are not supported. Generate it by yourself! \n" &&
        0
    );

        return 0;
    }
}

// Copide from Accel-sim cache_config::hash_function
unsigned GCoM::HashAddress(Address addr, unsigned nBins, unsigned offSetBits, EHashFunction hashFunctionType)
{
    unsigned log2nBins = LogB2(nBins);

    unsigned binIndex = 0;

    switch (hashFunctionType)
    {
    case EHashFunction::FERMI_HASH_SET_FUNCTION:
    {
        /*
         * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
         * Distance Theory" Cedric Nugteren et al. HPCA 2014
         */
        unsigned lowerXor = 0;
        unsigned upperXor = 0;

        if (nBins == 32 || nBins == 64)
        {
            // Lower xor value is bits 7-11
            lowerXor = (addr >> offSetBits) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            upperXor = (addr & 0xE000) >> 13;   // Bits 13, 14, 15
            upperXor |= (addr & 0x20000) >> 14; // Bit 17
            upperXor |= (addr & 0x80000) >> 15; // Bit 19

            binIndex = (lowerXor ^ upperXor);

            // 48KB cache prepends the binIndex with bit 12
            if (nBins == 64)
                binIndex |= (addr & 0x1000) >> 7;
        }
        else
        {
            assert(false && "Incorrect number of bins for Fermi hashing function.\n");
        }
        break;
    }
    case EHashFunction::BITWISE_XORING_FUNCTION:
    {
        Address higherBits = addr >> (offSetBits + log2nBins);
        unsigned index = (addr >> offSetBits) & (nBins - 1);
        binIndex = (index) ^ (higherBits & (nBins - 1));
        break;
    }
    case EHashFunction::HASH_IPOLY_FUNCTION:
    {
        Address higherBits = addr >> (offSetBits + log2nBins);
        unsigned index = (addr >> offSetBits) & (nBins - 1);
        binIndex = HashAdressWithIpolyFunction(higherBits, index, nBins);
        break;
    }

    case EHashFunction::LINEAR_SET_FUNCTION:
    {
        binIndex = (addr >> offSetBits) & (nBins - 1);
        break;
    }

    default:
    {
        assert(false && "\nUndefined hash function.\n");
        break;
    }
    }

    assert(binIndex < nBins);

    return binIndex;
}