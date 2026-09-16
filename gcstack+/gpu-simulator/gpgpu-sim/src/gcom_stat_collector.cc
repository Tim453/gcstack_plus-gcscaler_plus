#include <vector>
#include <cmath>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include "gcom_stat_collector.h"
#include "../libcuda/gpgpu_context.h"
#include "abstract_hardware_model.h"

using namespace std;

// Data struct shared with GCoM project
// Version history:
//   v0: original format (unsigned hit/miss counters)
//   v1: added warpInstIdx, switched to uint8_t counters
//   v2: PC-based keying — dropped warpInstIdx, consumer matches by PC + FIFO order
struct WarpInstCacheStatPrint
{
  int pc;
  int op;
  unsigned warpInstIdx;  // only used in v1; ignored in v2
  uint8_t l1Hit;
  uint8_t l1Miss;
  uint8_t coalescedL1Miss;  // Miss to the same MSHR entry is coalesced
  uint8_t l2Hit;
  uint8_t l2Miss;
  std::vector<new_addr_type> accessQAddr;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & pc;
    ar & op;
    if (version >= 2)
    {
      // v2: PC-based keying. No warpInstIdx needed — consumer matches by PC + FIFO order.
      ar & l1Hit;
      ar & l1Miss;
      ar & coalescedL1Miss;
      ar & l2Hit;
      ar & l2Miss;
    }
    else if (version == 1)
    {
      ar & warpInstIdx;
      ar & l1Hit;
      ar & l1Miss;
      ar & coalescedL1Miss;
      ar & l2Hit;
      ar & l2Miss;
    } else if (version == 0)
    {
      unsigned v0L1Hit, v0L1Miss, v0CoalescedL1Miss, v0L2Hit, v0L2Miss;
      ar & v0L1Hit;
      ar & v0L1Miss;
      ar & v0CoalescedL1Miss;
      ar & v0L2Hit;
      ar & v0L2Miss;
      l1Hit = (uint8_t) v0L1Hit;
      l1Miss = (uint8_t) v0L1Miss;
      coalescedL1Miss = (uint8_t) v0CoalescedL1Miss;
      l2Hit = (uint8_t) v0L2Hit;
      l2Miss = (uint8_t) v0L2Miss;
    }
    ar & accessQAddr;
  }

  WarpInstCacheStatPrint &operator=(const GComStatCollector::WarpInstCacheStat &rhs)
  {
    pc = rhs.pc;
    op = rhs.op;
    warpInstIdx = rhs.warpInstIdx;
    l1Hit = rhs.l1Hit;
    l1Miss = rhs.l1Miss;
    coalescedL1Miss = rhs.coalescedL1Miss;
    l2Hit = rhs.l2Hit;
    l2Miss = rhs.l2Miss;
    accessQAddr = rhs.accessQAddr;
    return *this;
  }
};

BOOST_CLASS_VERSION(WarpInstCacheStatPrint, 2)

struct KernelStatPrint
{
  double totalCPI;
  double baseCPI;
  double comDataCPI;
  double comStructCPI;
  double memDataCPI;
  double memStructCPI;
  double syncCPI;
  double controlCPI;
  double emptyWarpSlotCPI;
  double idleSubcoreCPI;
  double idleSMCPI;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) 
  {
    ar & totalCPI;
    if (version >= 1)
    {
      ar & baseCPI;
      ar & comDataCPI;
      ar & comStructCPI;
      ar & memDataCPI;
      ar & memStructCPI;
      ar & syncCPI;
      ar & controlCPI;
      ar & emptyWarpSlotCPI;
      ar & idleSubcoreCPI;
      ar & idleSMCPI;
    }
  }

  KernelStatPrint &operator=(const GComStatCollector::KernelStat &rhs)
  {
    totalCPI = rhs.totalCPI;
    baseCPI = rhs.baseCPI;
    comDataCPI = rhs.comDataCPI;
    comStructCPI = rhs.comStructCPI;
    memDataCPI = rhs.memDataCPI;
    memStructCPI = rhs.memStructCPI;
    syncCPI = rhs.syncCPI;
    controlCPI = rhs.controlCPI;
    emptyWarpSlotCPI = rhs.emptyWarpSlotCPI;
    idleSubcoreCPI = rhs.idleSubcoreCPI;
    idleSMCPI = rhs.idleSMCPI;
    return *this;
  }
};

BOOST_CLASS_VERSION(KernelStatPrint, 1)

// Data struct shared with GCoM project
struct CacheStatHeaderPrint
{
  int kernelNumber;
  std::streamoff nextKernelOffset;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) 
  {
    ar & kernelNumber;
    ar & nextKernelOffset;
  }
};

GComStatCollector::GComStatCollector(gpgpu_context *ctx)
{
  mCtx = ctx;
}

void GComStatCollector::BeginKernel(kernel_info_t *kernelInfo) 
{
  int nCTA = (int) kernelInfo->num_blocks();
  double warpSize = 32; // Hardcoded!
  mNumWarpPerCTA = ceil((double) kernelInfo->threads_per_cta() / warpSize);

  vector<vector<WarpInstCacheStat> > (nCTA * mNumWarpPerCTA).swap(mKernelCacheStat);
  vector<unsigned> (nCTA * mNumWarpPerCTA).swap(mWarpInstHeadIdx);
}

void GComStatCollector::AddInstAccessq(warp_inst_t &warpInst, unsigned ctaId, unsigned dynamicWarpId)
{
  int wid = dynamicWarpId;
  int warpIdx = ctaId * mNumWarpPerCTA + wid;

  // Track memory operations (LOAD_OP, STORE_OP, etc.)
  // warpInstIdx is a sequential memory-instruction-only counter, so it stays
  // synchronized with GCScaler's per-warp memory instruction index regardless
  // of extra non-memory instructions in the enhanced trace (NOPs, DEPBARs, etc.)
  if (warpInst.is_load() || warpInst.is_store())
  {
    WarpInstCacheStat warpInstCacheStat;
    warpInstCacheStat.pc = warpInst.pc;
    warpInstCacheStat.op = static_cast<int>(warpInst.op);
    warpInstCacheStat.warpInstIdx = mWarpInstHeadIdx[warpIdx];
    warpInstCacheStat.issueCycle = warpInst.get_issuecycle();

    warpInstCacheStat.accessQAddr.resize(warpInst.accessq_count());
    for (int j = 0; j < warpInst.accessq_count(); j++)
    {
      mem_access_t access = warpInst.accessq_iter(j);
      new_addr_type addr = access.get_addr();
      warpInstCacheStat.accessQAddr[j] = addr;
    }
    #pragma omp critical
    {
      mKernelCacheStat[warpIdx].push_back(warpInstCacheStat);
    }
    #pragma omp atomic
    mWarpInstHeadIdx[warpIdx] += 1;
  }
}

void GComStatCollector::StatMemAccess(const warp_inst_t &warpInst, unsigned ctaId, unsigned dynamicWarpId, EMemStatus status)
{
  int wid = dynamicWarpId;
  int warpIdx = ctaId * mNumWarpPerCTA + wid;
  
  WarpInstCacheStat *matched = NULL;
  vector<WarpInstCacheStat> &warpCacheStat = mKernelCacheStat[warpIdx];

  if(warpCacheStat.size() == 0)
    return;

  for ( int i = warpCacheStat.size() - 1; i >=0; i--)
  {
    if (warpCacheStat[i].issueCycle == warpInst.get_issuecycle()
        && warpCacheStat[i].pc == warpInst.pc)
    {
      matched = &warpCacheStat[i];
      break;
    }
  }

  assert(matched != NULL);

  #pragma omp critical
  {
  switch (status) {
    case EMemStatus::l1Hit:
      matched->l1Hit++;
      break;
    case EMemStatus::l1Miss:
      matched->l1Miss++;
      matched->coalescedL1Miss++;
      break;
    case EMemStatus::l1HitReserved:
      matched->l1Miss++;
      break;
    case EMemStatus::l2Hit:
      matched->l2Hit++;
      break;
    case EMemStatus::l2Miss:
      matched->l2Miss++;
      break;
    default:
      abort();
  }
  }
}

void GComStatCollector::EndKernel()
{
  mKernelNumber++;
  
  KernelStat kernelStat;

  // Calculate Kernel CPI
  unsigned long long totCycle = mCtx->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle;
  unsigned long long threadInsn = mCtx->the_gpgpusim->g_the_gpu->gpu_tot_sim_insn;
  unsigned long long kernelThreadInsn = threadInsn - mPrevKernelStats.threadInsn;
  double kernelCpi = ((double) totCycle - mPrevKernelStats.totCycle) / kernelThreadInsn;
  mPrevKernelStats.totCycle = totCycle;
  mPrevKernelStats.threadInsn = threadInsn;
  kernelStat.totalCPI = kernelCpi;

  // Generate per kernel GCStack CPI stacks
  shader_core_stats *allSMStatPtr = mCtx->the_gpgpusim->g_the_gpu->get_shader_stats();
  unsigned nSM = mCtx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->num_shader();
  unsigned nSubcore = mCtx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->gpgpu_num_sched_per_core;
  unsigned nSMMaxWarp = mCtx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->max_warps_per_shader;

  GCStackCycleBreakdown accumulatedCycleBreakdown;
  GCStackGen(allSMStatPtr->GCStack_CPI, allSMStatPtr->shader_cycles, allSMStatPtr->lazy_struct[0], allSMStatPtr->lazy_struct[1], 
      (double) allSMStatPtr->emptyWarpSlotValue, (double) allSMStatPtr->idleSubCoreValue, (double) allSMStatPtr->idleSMValue, 
      nSM, (double) nSubcore, nSMMaxWarp, accumulatedCycleBreakdown);
  
  GCStackCycleBreakdown kernelCycleBreakdown;
  kernelCycleBreakdown.base = accumulatedCycleBreakdown.base - mPrevKernelStats.baseCycle;
  kernelCycleBreakdown.memStruct = accumulatedCycleBreakdown.memStruct - mPrevKernelStats.memStructCycle;
  kernelCycleBreakdown.memData = accumulatedCycleBreakdown.memData - mPrevKernelStats.memDataCycle;
  kernelCycleBreakdown.sync = accumulatedCycleBreakdown.sync - mPrevKernelStats.syncCycle;
  kernelCycleBreakdown.comStruct = accumulatedCycleBreakdown.comStruct - mPrevKernelStats.comStructCycle;
  kernelCycleBreakdown.comData = accumulatedCycleBreakdown.comData - mPrevKernelStats.comDataCycle;
  kernelCycleBreakdown.control = accumulatedCycleBreakdown.control - mPrevKernelStats.controlCycle;
  kernelCycleBreakdown.emptyWarpSlot = accumulatedCycleBreakdown.emptyWarpSlot - mPrevKernelStats.emptyWarpSlotCycle;
  kernelCycleBreakdown.idleSubcore = accumulatedCycleBreakdown.idleSubcore - mPrevKernelStats.subcoreIdleCycle;
  kernelCycleBreakdown.idleSM = accumulatedCycleBreakdown.idleSM - mPrevKernelStats.smIdleCycle;

  kernelStat.baseCPI = kernelCycleBreakdown.base / kernelThreadInsn;
  kernelStat.comDataCPI = kernelCycleBreakdown.comData / kernelThreadInsn;
  kernelStat.comStructCPI = kernelCycleBreakdown.comStruct / kernelThreadInsn;
  kernelStat.memDataCPI = kernelCycleBreakdown.memData / kernelThreadInsn;
  kernelStat.memStructCPI = kernelCycleBreakdown.memStruct / kernelThreadInsn;
  kernelStat.syncCPI = kernelCycleBreakdown.sync / kernelThreadInsn;
  kernelStat.controlCPI = kernelCycleBreakdown.control / kernelThreadInsn;
  kernelStat.emptyWarpSlotCPI = kernelCycleBreakdown.emptyWarpSlot / kernelThreadInsn;
  kernelStat.idleSubcoreCPI = kernelCycleBreakdown.idleSubcore / kernelThreadInsn;
  kernelStat.idleSMCPI = kernelCycleBreakdown.idleSM / kernelThreadInsn;

  mPrevKernelStats.baseCycle = accumulatedCycleBreakdown.base;
  mPrevKernelStats.comDataCycle = accumulatedCycleBreakdown.comData;
  mPrevKernelStats.comStructCycle = accumulatedCycleBreakdown.comStruct;
  mPrevKernelStats.memDataCycle = accumulatedCycleBreakdown.memData;
  mPrevKernelStats.memStructCycle = accumulatedCycleBreakdown.memStruct;
  mPrevKernelStats.syncCycle = accumulatedCycleBreakdown.sync;
  mPrevKernelStats.controlCycle = accumulatedCycleBreakdown.control;
  mPrevKernelStats.emptyWarpSlotCycle = accumulatedCycleBreakdown.emptyWarpSlot;
  mPrevKernelStats.subcoreIdleCycle = accumulatedCycleBreakdown.idleSubcore;
  mPrevKernelStats.smIdleCycle = accumulatedCycleBreakdown.idleSM;

  if (abs(accumulatedCycleBreakdown.total - totCycle) > 5)
    cout << "[Error] GCStack total cycle mismatch: " << accumulatedCycleBreakdown.total << " vs " << totCycle << endl;

  mKernelStats.push_back(kernelStat);

  // Set output file identifier
  mOutputFileIdentifier = "modern";

  ExportKernelGCoMStat();
}

void GComStatCollector::ExportKernelGCoMStat()
{
  // Prepare to export cache stat data
  vector<vector<WarpInstCacheStatPrint> > kernelCacheStat;
  for (int i = 0; i < mKernelCacheStat.size(); i++)
  {
    vector<WarpInstCacheStatPrint> warpCacheStat;
    for (int j = 0; j < mKernelCacheStat[i].size(); j++)
    {
      WarpInstCacheStatPrint warpInstCacheStat;
      warpInstCacheStat = mKernelCacheStat[i][j];
      warpCacheStat.push_back(warpInstCacheStat);
    }
    kernelCacheStat.push_back(warpCacheStat);
  }

  // append cache stat to binary file
  string filename = "./accelsim_cache_stat_" + mOutputFileIdentifier + ".bin";
  ofstream writeFile;
  if (mKernelNumber == 1)
    writeFile.open(filename, ios::binary);
  else
    writeFile.open(filename, ios::binary | ios::app);

  boost::archive::binary_oarchive oa(writeFile);
  CacheStatHeaderPrint header;
  header.kernelNumber = mKernelNumber;
  oa & header;
  oa & kernelCacheStat;
  writeFile.close();

  // prepare to export kernel stat
  vector<KernelStatPrint> kernelStats;
  for (int i = 0; i < mKernelStats.size(); i++)
  {
    KernelStatPrint kernelStatPrint;
    kernelStatPrint = mKernelStats[i];
    kernelStats.push_back(kernelStatPrint);
  }

  // overwrite kernel CPI binary file
  filename = "./kernel_cpi_" + mOutputFileIdentifier + ".bin";
  writeFile.open(filename, ios::binary);
  boost::archive::binary_oarchive oa2(writeFile);
  oa2 & kernelStats;
  writeFile.close();
}