#ifndef _INST_DECODER_H
#define _INST_DECODER_H

#include <vector>
#include <unordered_map>
#include <filesystem>
#include "instruction.h"
#include "trace_parser.h"
#include "trace_parser_legacy.h"
#include "traced_execution.h"

namespace GCoM
{
    // SASS decoder dependent on SASS trace, but independent of hardware configuration
    class SASSDecoder
    {
    public:
        SASSDecoder(OptionParser opp);
        ~SASSDecoder();

        // input: mAccelsimTracer
        // output: mWarps, kernelInfo
        int ParseKernalTrace();

        WarpInst &GetWarpInst(int warpId, int instIdx); // This will also used to update cache stats in warp inst
        Warp &GetWarp(int warpId);
        int FreeWarps(); // free warps after cache simulation and representative warp selection of a kernel

        KernelInfo mKernelInfo;

    private:
        std::vector<Warp> mWarps; // decoded warps for a kernel
        std::filesystem::path mSASSTracePath;
        bool m_use_enhanced_format;

        // Enhanced (protobuf) parser
        trace_parser *mAccelsimTracer;
        traced_execution m_static_trace_info;
        std::vector<trace_command> mCommandlist;
        std::vector<trace_command>::iterator mCurrentGPUCommand;

        // Legacy (text) parser
        legacy_trace::trace_parser *m_legacy_tracer;
        std::vector<legacy_trace::trace_command> m_legacy_commandlist;
        std::vector<legacy_trace::trace_command>::iterator m_legacy_current_command;

        // Input: instTrace, kernel_trace_info
        // Output: warpInst with decoded information, mKernelInfo.numThreadInsts
        int DecocdeInst(inst_trace_t &instTrace, kernel_trace_t kernel_trace_info, WarpInst &warpInst);
        int DecocdeInstLegacy(legacy_trace::inst_trace_t &instTrace, legacy_trace::kernel_trace_t kernel_trace_info, WarpInst &warpInst);
        int ClassifyLatencyInitIntv(std::string instTraceOpcode, WarpInst &warpInst);

        int ParseKernalTraceLegacy();
        int ParseKernalTraceEnhanced();

    }; // class SASSDecoder

    static const std::unordered_map<std::string, EUArchOp> BlackwellOpcodeMap = {
        {"FADD", EUArchOp::SP_OP},
        {"FADD32I", EUArchOp::SP_OP},
        {"FCHK", EUArchOp::SP_OP},
        {"FFMA32I", EUArchOp::SP_OP},
        {"FFMA", EUArchOp::SP_OP},
        {"FMNMX", EUArchOp::SP_OP},
        {"FMUL", EUArchOp::SP_OP},
        {"FMUL32I", EUArchOp::SP_OP},
        {"FSEL", EUArchOp::SP_OP},
        {"FSET", EUArchOp::SP_OP},
        {"FSETP", EUArchOp::SP_OP},
        {"FSWZADD", EUArchOp::SP_OP},
        // SFU
        {"MUFU", EUArchOp::SFU_OP},

        // Floating Point 16 Instructions
        {"HADD2", EUArchOp::SP_OP},
        {"HADD2_32I", EUArchOp::SP_OP},
        {"HFMA2", EUArchOp::SP_OP},
        {"HFMA2_32I", EUArchOp::SP_OP},
        {"HMUL2", EUArchOp::SP_OP},
        {"HMUL2_32I", EUArchOp::SP_OP},
        {"HSET2", EUArchOp::SP_OP},
        {"HSETP2", EUArchOp::SP_OP},
        {"HMNMX2", EUArchOp::SP_OP},

        // Tensor Core Instructions
        // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
        {"HMMA", EUArchOp::TENSOR_CORE_OP},
        {"DMMA", EUArchOp::DP_OP},
        {"BMMA", EUArchOp::TENSOR_CORE_OP},
        {"IMMA", EUArchOp::TENSOR_CORE_OP},

        // Double Point Instructions
        {"DADD", EUArchOp::DP_OP},
        {"DFMA", EUArchOp::DP_OP},
        {"DMUL", EUArchOp::DP_OP},
        {"DSETP", EUArchOp::DP_OP},

        // Integer Instructions
        {"BMSK", EUArchOp::INTP_OP},
        {"BREV", EUArchOp::INTP_OP},
        {"FLO", EUArchOp::INTP_OP},
        {"IABS", EUArchOp::INTP_OP},
        {"IADD", EUArchOp::INTP_OP},
        {"IADD3", EUArchOp::INTP_OP},
        {"IADD32I", EUArchOp::INTP_OP},
        {"IDP", EUArchOp::INTP_OP},
        {"IDP4A", EUArchOp::INTP_OP},
        {"IMAD", EUArchOp::SP_OP},
        {"IMNMX", EUArchOp::INTP_OP},
        {"IMUL", EUArchOp::INTP_OP},
        {"IMUL32I", EUArchOp::INTP_OP},
        {"ISCADD", EUArchOp::INTP_OP},
        {"ISCADD32I", EUArchOp::INTP_OP},
        {"ISETP", EUArchOp::INTP_OP},
        {"LEA", EUArchOp::INTP_OP},
        {"LOP", EUArchOp::INTP_OP},
        {"LOP3", EUArchOp::INTP_OP},
        {"LOP32I", EUArchOp::INTP_OP},
        {"POPC", EUArchOp::INTP_OP},
        {"SHF", EUArchOp::INTP_OP},
        {"SHL", EUArchOp::INTP_OP}, //////////
        {"SHR", EUArchOp::INTP_OP},
        {"VABSDIFF", EUArchOp::INTP_OP},
        {"VABSDIFF4", EUArchOp::INTP_OP},
        {"VIMNMX", EUArchOp::INTP_OP},
        {"VIMNMX3", EUArchOp::INTP_OP},

        // Conversion Instructions
        {"F2FP", EUArchOp::SFU_OP},
        {"F2F", EUArchOp::DP_OP},
        {"F2I", EUArchOp::SFU_OP},
        {"I2F", EUArchOp::SFU_OP},
        {"I2I", EUArchOp::SFU_OP},
        {"I2IP", EUArchOp::SFU_OP},
        {"I2FP", EUArchOp::SFU_OP},
        {"F2IP", EUArchOp::SFU_OP},
        {"FRND", EUArchOp::SFU_OP},

        // Movement Instructions
        {"MOV", EUArchOp::INTP_OP},
        {"MOV32I", EUArchOp::INTP_OP},
        {"MOVM", EUArchOp::TENSOR_CORE_OP}, // move matrix
        {"PRMT", EUArchOp::INTP_OP},
        {"SEL", EUArchOp::INTP_OP},
        {"SGXT", EUArchOp::INTP_OP},
        {"SHFL", EUArchOp::ALU_OP},

        // Predicate Instructions
        {"PLOP3", EUArchOp::ALU_OP},
        {"PSETP", EUArchOp::ALU_OP},
        {"P2R", EUArchOp::ALU_OP},
        {"R2P", EUArchOp::ALU_OP},

        // Load/Store Instructions
        {"LD", EUArchOp::LOAD_OP},
        // For now, we ignore constant loads, consider it as ALU_OP, TO DO
        {"LDC", EUArchOp::LOAD_OP},
        {"LDCU", EUArchOp::LOAD_OP},
        {"LDG", EUArchOp::LOAD_OP},
        {"LDL", EUArchOp::LOAD_OP},
        {"LDS", EUArchOp::LOAD_OP},
        {"LDSM", EUArchOp::LOAD_OP}, //
        {"ST", EUArchOp::STORE_OP},
        {"STG", EUArchOp::STORE_OP},
        {"STL", EUArchOp::STORE_OP},
        {"STS", EUArchOp::STORE_OP},
        {"MATCH", EUArchOp::ALU_OP},
        {"QSPC", EUArchOp::ALU_OP},
        {"ATOM", EUArchOp::LOAD_OP},
        {"ATOMS", EUArchOp::LOAD_OP},
        {"ATOMG", EUArchOp::LOAD_OP},
        {"RED", EUArchOp::STORE_OP},
        {"REDG", EUArchOp::STORE_OP},
        {"CCTL", EUArchOp::ALU_OP},
        {"CCTLL", EUArchOp::ALU_OP},
        {"ERRBAR", EUArchOp::ALU_OP},
        {"MEMBAR", EUArchOp::MEMORY_BARRIER_OP},
        {"CGAERRBAR", EUArchOp::ALU_OP},
        {"CGABAR", EUArchOp::ALU_OP},
        {"CGABAR_ARV", EUArchOp::ALU_OP},
        {"CGABAR_WAIT", EUArchOp::ALU_OP},
        {"UCGABAR_ARV", EUArchOp::ALU_OP},
        {"UCGABAR_WAIT", EUArchOp::ALU_OP},
        {"CCTLT", EUArchOp::ALU_OP},

        {"LDGDEPBAR", EUArchOp::LOAD_OP},
        {"LDGSTS", EUArchOp::LOAD_OP},

        // Uniform Datapath Instruction
        // UDP unit
        // for more info about UDP, see
        // https://www.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
        {"R2UR", EUArchOp::INTP_OP},
        {"S2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBMSK", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBREV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UCLEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UF2FP", EUArchOp::SFU_OP},
        {"UFLO", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIADD3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIMAD", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UISETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULDC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP32I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UMOV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UP2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPLOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPOPC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPRMT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPSETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UR2UP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USEL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USGXT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHF", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UVIMNMX", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFADD", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFMUL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UF2I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2F", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2FP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2IP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFFMA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFRND", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"VOTEU", EUArchOp::SPECIALIZED_UNIT_4_OP},

        // Texture Instructions
        // For now, we ignore texture loads, consider it as ALU_OP
        {"TEX", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD4", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TMML", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXQ", EUArchOp::SPECIALIZED_UNIT_2_OP},

        // Surface Instructions //
        {"SUATOM", EUArchOp::ALU_OP},
        {"SULD", EUArchOp::ALU_OP},
        {"SUQUERY", EUArchOp::ALU_OP},
        {"SURED", EUArchOp::ALU_OP},
        {"SUST", EUArchOp::ALU_OP},

        // Control Instructions
        // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
        {"BMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BPT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRA", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BREAK", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRXU", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSSY", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"CALL", EUArchOp::CALL_OPS},
        {"EXIT", EUArchOp::EXIT_OPS},
        {"JMP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMXU", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"KILL", EUArchOp::SPECIALIZED_UNIT_3_OP},
        {"NANOSLEEP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RET", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RPCMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RTT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"WARPSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"YIELD", EUArchOp::SPECIALIZED_UNIT_1_OP},

        // Miscellaneous Instructions
        {"REDUX", EUArchOp::ALU_OP},
        {"B2R", EUArchOp::ALU_OP},
        {"BAR", EUArchOp::BARRIER_OP},
        {"CS2R", EUArchOp::INTP_OP},
        {"CS2UR", EUArchOp::INTP_OP},
        {"CSMTEST", EUArchOp::ALU_OP},
        {"DEPBAR", EUArchOp::ALU_OP},
        {"GETLMEMBASE", EUArchOp::ALU_OP},
        {"LEPC", EUArchOp::ALU_OP},
        {"NOP", EUArchOp::NO_OP},
        {"PMTRIG", EUArchOp::ALU_OP},
        {"R2B", EUArchOp::ALU_OP},
        {"S2R", EUArchOp::ALU_OP},
        {"SETCTAID", EUArchOp::ALU_OP},
        {"SETLMEMBASE", EUArchOp::ALU_OP},
        {"VOTE", EUArchOp::ALU_OP},
        {"VOTE_VTG", EUArchOp::ALU_OP},
    };

    // Based on https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#hopper-instruction-set
    static const std::unordered_map<std::string, EUArchOp> HopperOpcodeMap = {
        // Floating Point 32 Instructions
        {"FADD", EUArchOp::SP_OP},
        {"FADD32I", EUArchOp::SP_OP},
        {"FCHK", EUArchOp::SP_OP},
        {"FFMA32I", EUArchOp::SP_OP},
        {"FFMA", EUArchOp::SP_OP},
        {"FMNMX", EUArchOp::SP_OP},
        {"FMUL", EUArchOp::SP_OP},
        {"FMUL32I", EUArchOp::SP_OP},
        {"FSEL", EUArchOp::SP_OP},
        {"FSET", EUArchOp::SP_OP},
        {"FSETP", EUArchOp::SP_OP},
        {"FSWZADD", EUArchOp::SP_OP},
        // SFU
        {"MUFU", EUArchOp::SFU_OP},

        // Floating Point 16 Instructions
        {"HADD2", EUArchOp::SP_OP},
        {"HADD2_32I", EUArchOp::SP_OP},
        {"HFMA2", EUArchOp::SP_OP},
        {"HFMA2_32I", EUArchOp::SP_OP},
        {"HMUL2", EUArchOp::SP_OP},
        {"HMUL2_32I", EUArchOp::SP_OP},
        {"HSET2", EUArchOp::SP_OP},
        {"HSETP2", EUArchOp::SP_OP},
        {"HMNMX2", EUArchOp::SP_OP},

        // Tensor Core Instructions
        {"HMMA", EUArchOp::TENSOR_CORE_OP},
        {"DMMA", EUArchOp::DP_OP},
        {"BMMA", EUArchOp::TENSOR_CORE_OP},
        {"IMMA", EUArchOp::TENSOR_CORE_OP},
        // Warpgroup MMA (Hopper-specific asynchronous tensor core ops)
        {"HGMMA", EUArchOp::TENSOR_CORE_OP},
        {"IGMMA", EUArchOp::TENSOR_CORE_OP},
        {"QGMMA", EUArchOp::TENSOR_CORE_OP},
        {"BGMMA", EUArchOp::TENSOR_CORE_OP},
        {"WARPGROUP", EUArchOp::ALU_OP},
        {"WARPGROUPSET", EUArchOp::ALU_OP},

        // Double Point Instructions
        {"DADD", EUArchOp::DP_OP},
        {"DFMA", EUArchOp::DP_OP},
        {"DMUL", EUArchOp::DP_OP},
        {"DSETP", EUArchOp::DP_OP},

        // Integer Instructions
        {"BMSK", EUArchOp::INTP_OP},
        {"BREV", EUArchOp::INTP_OP},
        {"FLO", EUArchOp::INTP_OP},
        {"IABS", EUArchOp::INTP_OP},
        {"IADD", EUArchOp::INTP_OP},
        {"IADD3", EUArchOp::INTP_OP},
        {"IADD32I", EUArchOp::INTP_OP},
        {"IDP", EUArchOp::INTP_OP},
        {"IDP4A", EUArchOp::INTP_OP},
        {"IMAD", EUArchOp::SP_OP},
        {"IMNMX", EUArchOp::INTP_OP},
        {"IMUL", EUArchOp::INTP_OP},
        {"IMUL32I", EUArchOp::INTP_OP},
        {"ISCADD", EUArchOp::INTP_OP},
        {"ISCADD32I", EUArchOp::INTP_OP},
        {"ISETP", EUArchOp::INTP_OP},
        {"LEA", EUArchOp::INTP_OP},
        {"LOP", EUArchOp::INTP_OP},
        {"LOP3", EUArchOp::INTP_OP},
        {"LOP32I", EUArchOp::INTP_OP},
        {"POPC", EUArchOp::INTP_OP},
        {"SHF", EUArchOp::INTP_OP},
        {"SHL", EUArchOp::INTP_OP},
        {"SHR", EUArchOp::INTP_OP},
        {"VABSDIFF", EUArchOp::INTP_OP},
        {"VABSDIFF4", EUArchOp::INTP_OP},
        {"VIMNMX", EUArchOp::INTP_OP},
        {"VIMNMX3", EUArchOp::INTP_OP},
        {"VHMNMX", EUArchOp::INTP_OP},
        {"VIADD", EUArchOp::INTP_OP},
        {"VIADDMNMX", EUArchOp::INTP_OP},

        // Conversion Instructions
        {"F2FP", EUArchOp::SFU_OP},
        {"F2F", EUArchOp::DP_OP},
        {"F2I", EUArchOp::SFU_OP},
        {"I2F", EUArchOp::SFU_OP},
        {"I2I", EUArchOp::SFU_OP},
        {"I2IP", EUArchOp::SFU_OP},
        {"I2FP", EUArchOp::SFU_OP},
        {"F2IP", EUArchOp::SFU_OP},
        {"FRND", EUArchOp::SFU_OP},

        // Movement Instructions
        {"MOV", EUArchOp::INTP_OP},
        {"MOV32I", EUArchOp::INTP_OP},
        {"MOVM", EUArchOp::TENSOR_CORE_OP}, // move matrix
        {"PRMT", EUArchOp::INTP_OP},
        {"SEL", EUArchOp::INTP_OP},
        {"SGXT", EUArchOp::INTP_OP},
        {"SHFL", EUArchOp::ALU_OP},

        // Predicate Instructions
        {"PLOP3", EUArchOp::ALU_OP},
        {"PSETP", EUArchOp::ALU_OP},
        {"P2R", EUArchOp::ALU_OP},
        {"R2P", EUArchOp::ALU_OP},

        // Load/Store Instructions
        {"LD", EUArchOp::LOAD_OP},
        {"LDC", EUArchOp::LOAD_OP},
        {"LDCU", EUArchOp::LOAD_OP},
        {"LDG", EUArchOp::LOAD_OP},
        {"LDL", EUArchOp::LOAD_OP},
        {"LDS", EUArchOp::LOAD_OP},
        {"LDSM", EUArchOp::LOAD_OP},
        {"LDGMC", EUArchOp::LOAD_OP},
        {"ST", EUArchOp::STORE_OP},
        {"STG", EUArchOp::STORE_OP},
        {"STL", EUArchOp::STORE_OP},
        {"STS", EUArchOp::STORE_OP},
        {"STSM", EUArchOp::STORE_OP},
        {"MATCH", EUArchOp::ALU_OP},
        {"QSPC", EUArchOp::ALU_OP},
        {"ATOM", EUArchOp::LOAD_OP},
        {"ATOMS", EUArchOp::LOAD_OP},
        {"ATOMG", EUArchOp::LOAD_OP},
        {"RED", EUArchOp::STORE_OP},
        {"REDG", EUArchOp::STORE_OP},
        {"CCTL", EUArchOp::ALU_OP},
        {"CCTLL", EUArchOp::ALU_OP},
        {"ERRBAR", EUArchOp::ALU_OP},
        {"MEMBAR", EUArchOp::MEMORY_BARRIER_OP},
        {"FENCE", EUArchOp::MEMORY_BARRIER_OP},
        {"CGAERRBAR", EUArchOp::ALU_OP},
        {"CCTLT", EUArchOp::ALU_OP},

        {"LDGDEPBAR", EUArchOp::LOAD_OP},
        {"LDGSTS", EUArchOp::LOAD_OP},

        // Tensor Memory Access Instructions (Hopper TMA)
        {"UBLKCP", EUArchOp::LOAD_OP},
        {"UBLKPF", EUArchOp::LOAD_OP},
        {"UBLKRED", EUArchOp::STORE_OP},
        {"UTMACCTL", EUArchOp::ALU_OP},
        {"UTMACMDFLUSH", EUArchOp::ALU_OP},
        {"UTMALDG", EUArchOp::LOAD_OP},
        {"UTMAPF", EUArchOp::LOAD_OP},
        {"UTMAREDG", EUArchOp::STORE_OP},
        {"UTMASTG", EUArchOp::STORE_OP},

        // Uniform Datapath Instruction
        {"R2UR", EUArchOp::INTP_OP},
        {"S2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBMSK", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBREV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UCLEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UF2FP", EUArchOp::SFU_OP},
        {"UFLO", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIADD3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIMAD", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UISETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULDC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP32I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UMOV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UP2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPLOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPOPC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPRMT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPSETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UR2UP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USEL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USGXT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHF", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UVIMNMX", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFADD", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFMUL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UF2I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2F", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2FP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UI2IP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFFMA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFRND", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"VOTEU", EUArchOp::SPECIALIZED_UNIT_4_OP},

        // Texture Instructions
        {"TEX", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD4", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TMML", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXQ", EUArchOp::SPECIALIZED_UNIT_2_OP},

        // Surface Instructions
        {"SUATOM", EUArchOp::ALU_OP},
        {"SULD", EUArchOp::ALU_OP},
        {"SUQUERY", EUArchOp::ALU_OP},
        {"SURED", EUArchOp::ALU_OP},
        {"SUST", EUArchOp::ALU_OP},

        // Control Instructions
        {"ACQBULK", EUArchOp::BARRIER_OP},
        {"BMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BPT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRA", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BREAK", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRXU", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSSY", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"CALL", EUArchOp::CALL_OPS},
        {"ELECT", EUArchOp::ALU_OP},
        {"ENDCOLLECTIVE", EUArchOp::ALU_OP},
        {"EXIT", EUArchOp::EXIT_OPS},
        {"JMP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMXU", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"KILL", EUArchOp::SPECIALIZED_UNIT_3_OP},
        {"NANOSLEEP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"PREEXIT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RET", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RPCMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RTT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"WARPSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"YIELD", EUArchOp::SPECIALIZED_UNIT_1_OP},

        // Miscellaneous Instructions
        {"REDUX", EUArchOp::ALU_OP},
        {"B2R", EUArchOp::ALU_OP},
        {"BAR", EUArchOp::BARRIER_OP},
        {"CS2R", EUArchOp::INTP_OP},
        {"CS2UR", EUArchOp::INTP_OP},
        {"CSMTEST", EUArchOp::ALU_OP},
        {"DEPBAR", EUArchOp::ALU_OP},
        {"GETLMEMBASE", EUArchOp::ALU_OP},
        {"LEPC", EUArchOp::ALU_OP},
        {"NOP", EUArchOp::NO_OP},
        {"PMTRIG", EUArchOp::ALU_OP},
        {"R2B", EUArchOp::ALU_OP},
        {"S2R", EUArchOp::ALU_OP},
        {"SETCTAID", EUArchOp::ALU_OP},
        {"SETLMEMBASE", EUArchOp::ALU_OP},
        {"VOTE", EUArchOp::ALU_OP},
        {"VOTE_VTG", EUArchOp::ALU_OP},
    };

    // Modified based on Accel-sim gpu-simulator/ISA_Def/turing_opcode.h
    static const std::unordered_map<std::string, EUArchOp> TuringOpcodeMap = {
        // Floating Point 32 Instructions
        {"FADD", EUArchOp::SP_OP},
        {"FADD32I", EUArchOp::SP_OP},
        {"FCHK", EUArchOp::SP_OP},
        {"FFMA32I", EUArchOp::SP_OP},
        {"FFMA", EUArchOp::SP_OP},
        {"FMNMX", EUArchOp::SP_OP},
        {"FMUL", EUArchOp::SP_OP},
        {"FMUL32I", EUArchOp::SP_OP},
        {"FSEL", EUArchOp::SP_OP},
        {"FSET", EUArchOp::SP_OP},
        {"FSETP", EUArchOp::SP_OP},
        {"FSWZADD", EUArchOp::SP_OP},
        // SFU
        {"MUFU", EUArchOp::SFU_OP},

        // Floating Point 16 Instructions
        {"HADD2", EUArchOp::SP_OP},
        {"HADD2_32I", EUArchOp::SP_OP},
        {"HFMA2", EUArchOp::SP_OP},
        {"HFMA2_32I", EUArchOp::SP_OP},
        {"HMUL2", EUArchOp::SP_OP},
        {"HMUL2_32I", EUArchOp::SP_OP},
        {"HSET2", EUArchOp::SP_OP},
        {"HSETP2", EUArchOp::SP_OP},

        // Tensor Core Instructions
        // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
        {"HMMA", EUArchOp::SPECIALIZED_UNIT_3_OP},
        {"BMMA", EUArchOp::SPECIALIZED_UNIT_3_OP},
        {"IMMA", EUArchOp::SPECIALIZED_UNIT_3_OP},

        // Double Point Instructions
        {"DADD", EUArchOp::DP_OP},
        {"DFMA", EUArchOp::DP_OP},
        {"DMUL", EUArchOp::DP_OP},
        {"DSETP", EUArchOp::DP_OP},

        // Integer Instructions
        {"BMSK", EUArchOp::INTP_OP},
        {"BREV", EUArchOp::INTP_OP},
        {"FLO", EUArchOp::INTP_OP},
        {"IABS", EUArchOp::INTP_OP},
        {"IADD", EUArchOp::INTP_OP},
        {"IADD3", EUArchOp::INTP_OP},
        {"IADD32I", EUArchOp::INTP_OP},
        {"IDP", EUArchOp::INTP_OP},
        {"IDP4A", EUArchOp::INTP_OP},
        {"IMAD", EUArchOp::INTP_OP},
        {"IMNMX", EUArchOp::INTP_OP},
        {"IMUL", EUArchOp::INTP_OP},
        {"IMUL32I", EUArchOp::INTP_OP},
        {"ISCADD", EUArchOp::INTP_OP},
        {"ISCADD32I", EUArchOp::INTP_OP},
        {"ISETP", EUArchOp::INTP_OP},
        {"LEA", EUArchOp::INTP_OP},
        {"LOP", EUArchOp::INTP_OP},
        {"LOP3", EUArchOp::INTP_OP},
        {"LOP32I", EUArchOp::INTP_OP},
        {"POPC", EUArchOp::INTP_OP},
        {"SHF", EUArchOp::INTP_OP},
        {"SHL", EUArchOp::INTP_OP}, //////////
        {"SHR", EUArchOp::INTP_OP},
        {"VABSDIFF", EUArchOp::INTP_OP},
        {"VABSDIFF4", EUArchOp::INTP_OP},

        // Conversion Instructions
        {"F2F", EUArchOp::ALU_OP},
        {"F2I", EUArchOp::ALU_OP},
        {"I2F", EUArchOp::ALU_OP},
        {"I2I", EUArchOp::ALU_OP},
        {"I2IP", EUArchOp::ALU_OP},
        {"FRND", EUArchOp::ALU_OP},

        // Movement Instructions
        {"MOV", EUArchOp::ALU_OP},
        {"MOV32I", EUArchOp::ALU_OP},
        {"MOVM", EUArchOp::ALU_OP}, // move matrix
        {"PRMT", EUArchOp::ALU_OP},
        {"SEL", EUArchOp::ALU_OP},
        {"SGXT", EUArchOp::ALU_OP},
        {"SHFL", EUArchOp::ALU_OP},

        // Predicate Instructions
        {"PLOP3", EUArchOp::ALU_OP},
        {"PSETP", EUArchOp::ALU_OP},
        {"P2R", EUArchOp::ALU_OP},
        {"R2P", EUArchOp::ALU_OP},

        // Load/Store Instructions
        {"LD", EUArchOp::LOAD_OP},
        // For now, we ignore constant loads, consider it as ALU_OP, TO DO
        {"LDC", EUArchOp::ALU_OP},
        {"LDG", EUArchOp::LOAD_OP},
        {"LDL", EUArchOp::LOAD_OP},
        {"LDS", EUArchOp::LOAD_OP},
        {"LDSM", EUArchOp::LOAD_OP}, //
        {"ST", EUArchOp::STORE_OP},
        {"STG", EUArchOp::STORE_OP},
        {"STL", EUArchOp::STORE_OP},
        {"STS", EUArchOp::STORE_OP},
        {"MATCH", EUArchOp::ALU_OP},
        {"QSPC", EUArchOp::ALU_OP},
        {"ATOM", EUArchOp::LOAD_OP}, // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"ATOMS", EUArchOp::STORE_OP},
        {"ATOMG", EUArchOp::LOAD_OP}, // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"RED", EUArchOp::LOAD_OP},   // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"CCTL", EUArchOp::ALU_OP},
        {"CCTLL", EUArchOp::ALU_OP},
        {"ERRBAR", EUArchOp::ALU_OP},
        {"MEMBAR", EUArchOp::MEMORY_BARRIER_OP},
        {"CCTLT", EUArchOp::ALU_OP},

        // Uniform Datapath Instruction
        // UDP unit
        // for more info about UDP, see
        // https://www.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
        {"R2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"S2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBMSK", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UBREV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UCLEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UFLO", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIADD3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UIMAD", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UISETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULDC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULEA", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"ULOP32I", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UMOV", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UP2UR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPLOP3", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPOPC", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPRMT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UPSETP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"UR2UP", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USEL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USGXT", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHF", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHL", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"USHR", EUArchOp::SPECIALIZED_UNIT_4_OP},
        {"VOTEU", EUArchOp::SPECIALIZED_UNIT_4_OP},

        // Texture Instructions
        // For now, we ignore texture loads, consider it as ALU_OP
        {"TEX", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD4", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TMML", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXQ", EUArchOp::SPECIALIZED_UNIT_2_OP},

        // Surface Instructions //
        {"SUATOM", EUArchOp::ALU_OP},
        {"SULD", EUArchOp::ALU_OP},
        {"SURED", EUArchOp::ALU_OP},
        {"SUST", EUArchOp::ALU_OP},

        // Control Instructions
        // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
        {"BMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BPT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRA", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BREAK", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRXU", EUArchOp::SPECIALIZED_UNIT_1_OP}, //
        {"BSSY", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"CALL", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"EXIT", EUArchOp::EXIT_OPS},
        {"JMP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMXU", EUArchOp::SPECIALIZED_UNIT_1_OP}, ///
        {"KILL", EUArchOp::SPECIALIZED_UNIT_3_OP},
        {"NANOSLEEP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RET", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RPCMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RTT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"WARPSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"YIELD", EUArchOp::SPECIALIZED_UNIT_1_OP},

        // Miscellaneous Instructions
        {"B2R", EUArchOp::ALU_OP},
        {"BAR", EUArchOp::BARRIER_OP},
        {"CS2R", EUArchOp::ALU_OP},
        {"CSMTEST", EUArchOp::ALU_OP},
        {"DEPBAR", EUArchOp::ALU_OP},
        {"GETLMEMBASE", EUArchOp::ALU_OP},
        {"LEPC", EUArchOp::ALU_OP},
        {"NOP", EUArchOp::ALU_OP},
        {"PMTRIG", EUArchOp::ALU_OP},
        {"R2B", EUArchOp::ALU_OP},
        {"S2R", EUArchOp::ALU_OP},
        {"SETCTAID", EUArchOp::ALU_OP},
        {"SETLMEMBASE", EUArchOp::ALU_OP},
        {"VOTE", EUArchOp::ALU_OP},
        {"VOTE_VTG", EUArchOp::ALU_OP}};

    // Modified based on Accel-sim gpu-simulator/ISA_Def/volta_opcode.h
    static const std::unordered_map<std::string, EUArchOp> VoltaOpcodeMap = {
        // Floating Point 32 Instructions
        {"FADD", EUArchOp::SP_OP},
        {"FADD32I", EUArchOp::SP_OP},
        {"FCHK", EUArchOp::SP_OP},
        {"FFMA32I", EUArchOp::SP_OP},
        {"FFMA", EUArchOp::SP_OP},
        {"FMNMX", EUArchOp::SP_OP},
        {"FMUL", EUArchOp::SP_OP},
        {"FMUL32I", EUArchOp::SP_OP},
        {"FSEL", EUArchOp::SP_OP},
        {"FSET", EUArchOp::SP_OP},
        {"FSETP", EUArchOp::SP_OP},
        {"FSWZADD", EUArchOp::SP_OP},
        // SFU
        {"MUFU", EUArchOp::SFU_OP},

        // Floating Point 16 Instructions
        {"HADD2", EUArchOp::SP_OP},
        {"HADD2_32I", EUArchOp::SP_OP},
        {"HFMA2", EUArchOp::SP_OP},
        {"HFMA2_32I", EUArchOp::SP_OP},
        {"HMUL2", EUArchOp::SP_OP},
        {"HMUL2_32I", EUArchOp::SP_OP},
        {"HSET2", EUArchOp::SP_OP},
        {"HSETP2", EUArchOp::SP_OP},

        // Tensor Core Instructions
        // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
        {"HMMA", EUArchOp::SPECIALIZED_UNIT_3_OP},

        // Double Point Instructions
        {"DADD", EUArchOp::DP_OP},
        {"DFMA", EUArchOp::DP_OP},
        {"DMUL", EUArchOp::DP_OP},
        {"DSETP", EUArchOp::DP_OP},

        // Integer Instructions
        {"BMSK", EUArchOp::INTP_OP},
        {"BREV", EUArchOp::INTP_OP},
        {"FLO", EUArchOp::INTP_OP},
        {"IABS", EUArchOp::INTP_OP},
        {"IADD", EUArchOp::INTP_OP},
        {"IADD3", EUArchOp::INTP_OP},
        {"IADD32I", EUArchOp::INTP_OP},
        {"IDP", EUArchOp::INTP_OP},
        {"IDP4A", EUArchOp::INTP_OP},
        {"IMAD", EUArchOp::INTP_OP},
        {"IMMA", EUArchOp::INTP_OP},
        {"IMNMX", EUArchOp::INTP_OP},
        {"IMUL", EUArchOp::INTP_OP},
        {"IMUL32I", EUArchOp::INTP_OP},
        {"ISCADD", EUArchOp::INTP_OP},
        {"ISCADD32I", EUArchOp::INTP_OP},
        {"ISETP", EUArchOp::INTP_OP},
        {"LEA", EUArchOp::INTP_OP},
        {"LOP", EUArchOp::INTP_OP},
        {"LOP3", EUArchOp::INTP_OP},
        {"LOP32I", EUArchOp::INTP_OP},
        {"POPC", EUArchOp::INTP_OP},
        {"SHF", EUArchOp::INTP_OP},
        {"SHR", EUArchOp::INTP_OP},
        {"VABSDIFF", EUArchOp::INTP_OP},
        {"VABSDIFF4", EUArchOp::INTP_OP},

        // Conversion Instructions
        {"F2F", EUArchOp::ALU_OP},
        {"F2I", EUArchOp::ALU_OP},
        {"I2F", EUArchOp::ALU_OP},
        {"I2I", EUArchOp::ALU_OP},
        {"I2IP", EUArchOp::ALU_OP},
        {"FRND", EUArchOp::ALU_OP},

        // Movement Instructions
        {"MOV", EUArchOp::ALU_OP},
        {"MOV32I", EUArchOp::ALU_OP},
        {"PRMT", EUArchOp::ALU_OP},
        {"SEL", EUArchOp::ALU_OP},
        {"SGXT", EUArchOp::ALU_OP},
        {"SHFL", EUArchOp::ALU_OP},

        // Predicate Instructions
        {"PLOP3", EUArchOp::ALU_OP},
        {"PSETP", EUArchOp::ALU_OP},
        {"P2R", EUArchOp::ALU_OP},
        {"R2P", EUArchOp::ALU_OP},

        // Load/Store Instructions
        {"LD", EUArchOp::LOAD_OP},
        // For now, we ignore constant loads, consider it as ALU_OP, TO DO
        {"LDC", EUArchOp::ALU_OP},
        {"LDG", EUArchOp::LOAD_OP},
        {"LDL", EUArchOp::LOAD_OP},
        {"LDS", EUArchOp::LOAD_OP},
        {"ST", EUArchOp::STORE_OP},
        {"STG", EUArchOp::STORE_OP},
        {"STL", EUArchOp::STORE_OP},
        {"STS", EUArchOp::STORE_OP},
        {"MATCH", EUArchOp::ALU_OP},
        {"QSPC", EUArchOp::ALU_OP},
        {"ATOM", EUArchOp::LOAD_OP}, // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"ATOMS", EUArchOp::STORE_OP},
        {"ATOMG", EUArchOp::LOAD_OP}, // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"RED", EUArchOp::LOAD_OP},   // Reference: Accelsim bool trace_warp_inst_t::parse_from_trace_struct
        {"CCTL", EUArchOp::ALU_OP},
        {"CCTLL", EUArchOp::ALU_OP},
        {"ERRBAR", EUArchOp::ALU_OP},
        {"MEMBAR", EUArchOp::MEMORY_BARRIER_OP},
        {"CCTLT", EUArchOp::ALU_OP},

        // Texture Instructions
        // For now, we ignore texture loads, consider it as ALU_OP
        {"TEX", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TLD4", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TMML", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXD", EUArchOp::SPECIALIZED_UNIT_2_OP},
        {"TXQ", EUArchOp::SPECIALIZED_UNIT_2_OP},

        // Control Instructions
        // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
        {"BMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BPT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRA", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BREAK", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BRX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSSY", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"BSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"CALL", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"EXIT", EUArchOp::EXIT_OPS},
        {"JMP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"JMX", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"KILL", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"NANOSLEEP", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RET", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RPCMOV", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"RTT", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"WARPSYNC", EUArchOp::SPECIALIZED_UNIT_1_OP},
        {"YIELD", EUArchOp::SPECIALIZED_UNIT_1_OP},

        // Miscellaneous Instructions
        {"B2R", EUArchOp::ALU_OP},
        {"BAR", EUArchOp::BARRIER_OP},
        {"CS2R", EUArchOp::ALU_OP},
        {"CSMTEST", EUArchOp::ALU_OP},
        {"DEPBAR", EUArchOp::ALU_OP},
        {"GETLMEMBASE", EUArchOp::ALU_OP},
        {"LEPC", EUArchOp::ALU_OP},
        {"NOP", EUArchOp::ALU_OP},
        {"PMTRIG", EUArchOp::ALU_OP},
        {"R2B", EUArchOp::ALU_OP},
        {"S2R", EUArchOp::ALU_OP},
        {"SETCTAID", EUArchOp::ALU_OP},
        {"SETLMEMBASE", EUArchOp::ALU_OP},
        {"VOTE", EUArchOp::ALU_OP},
        {"VOTE_VTG", EUArchOp::ALU_OP}};
}

#endif