// Removed gpgpu-sim dependency (gpgpu_sim* gpu parameter) since GCScaler does not
// use the full simulator. address_type is defined here directly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <bitset>

#include "trace.pb.h"
#include "instruction.pb.h"
#include "address.pb.h"

#ifndef TRACE_PARSER_H
#define TRACE_PARSER_H

const int WARP_SIZE = 32;

typedef unsigned long long address_type;

class traced_execution;

enum command_type {
  kernel_launch = 1,
  cpu_gpu_mem_copy,
  gpu_cpu_mem_copy,
};

enum address_space { GLOBAL_MEM = 1, SHARED_MEM, LOCAL_MEM, TEX_MEM };

enum address_scope {
  L1_CACHE = 1,
  L2_CACHE,
  SYS_MEM,
};

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

struct trace_command {
  std::string command_string;
  command_type m_type;
};

enum class TraceMemorySpace {
  STANDARD = 0,
  SHARED = 1,
  DSMEM = 2,      // Distributed shared memory (remote CTA within cluster)
  CONSTANT = 3,
  CALL_RET = 4
};

struct inst_memadd_info_t {
  uint64_t addrs[WARP_SIZE];
  int16_t width;
  uint32_t u_desc_value;
  TraceMemorySpace memory_space = TraceMemorySpace::STANDARD;

  void base_stride_decompress(unsigned long long base_address, int stride,
                              const std::bitset<WARP_SIZE> &mask);
  void base_delta_decompress(unsigned long long base_address,
                             const std::vector<long long> &deltas,
                             const std::bitset<WARP_SIZE> &mask);
};

struct inst_trace_t {
  inst_trace_t();
  inst_trace_t(const inst_trace_t &b);
  inst_trace_t(address_type pc, unsigned int unique_function_id, bool is_instruction_traced);

  bool is_instruction_traced;
  unsigned int m_pc;
  uint16_t m_unique_function_id;
  unsigned int m_next_traced_pc;
  unsigned int mask;
  std::string opcode;
  uint32_t block_idx_x;
  uint16_t block_idx_y;
  uint16_t block_idx_z;
  int32_t cluster_rank;  // rank of this CTA within its cluster (0-based)
  bool is_constant_addr_already_calculated;
  std::vector<std::unique_ptr<inst_memadd_info_t>> memadd_info;

  void parse_memref(unsigned int idx, unsigned int mem_width,
                    const std::bitset<WARP_SIZE> &mask_bits,
                    dynamic_trace::address addr_info);
  bool parse_from_pb(dynamic_trace::instruction pb_inst, unsigned tracer_version,
                     std::string kernel_name, traced_execution &static_trace_info);
  bool check_opcode_contain(const std::vector<std::string> &opcode,
                            std::string param) const;

  unsigned get_datawidth_from_opcode(
      const std::vector<std::string> &opcode) const;

  ~inst_trace_t();
};

struct traced_instructions_by_pc {
  traced_instructions_by_pc(address_type pc, unsigned int size_num_used_instructions)
      : pc(pc), num_traced_instructions(0) {
    num_used_instructions.resize(size_num_used_instructions, 0);
  }
  address_type pc;
  std::vector<inst_trace_t> instructions;
  unsigned int num_traced_instructions;
  std::vector<unsigned int> num_used_instructions;
};

struct kernel_trace_t {
  kernel_trace_t();

  std::string kernel_name;
  unsigned gpu_device_id;
  unsigned kernel_id;
  unsigned grid_dim_x;
  unsigned grid_dim_y;
  unsigned grid_dim_z;
  unsigned next_tb_to_parse_x;
  unsigned next_tb_to_parse_y;
  unsigned next_tb_to_parse_z;
  unsigned tb_dim_x;
  unsigned tb_dim_y;
  unsigned tb_dim_z;
  unsigned shmem;
  unsigned nregs;
  unsigned long cuda_stream_id;
  unsigned binary_verion;
  unsigned trace_verion;
  std::string nvbit_verion;
  unsigned long long shmem_base_addr;
  unsigned long long local_base_addr;
  unsigned int func_unique_id;
  bool is_cap_from_binary;
  unsigned cluster_dim_x;
  unsigned cluster_dim_y;
  unsigned cluster_dim_z;
};

class trace_parser {
 public:
  trace_parser(const char *kernellist_filepath, bool is_extra_trace_enabled,
               unsigned int kernel_id_filter_start,
               unsigned int kernel_id_filter_end);
  ~trace_parser() { dyn_trace.Clear(); }

  std::vector<trace_command> parse_commandlist_file();

  kernel_trace_t *parse_kernel_info(const std::string &kerneltraces_filepath,
                                    traced_execution &enhanced_trace_info);

  void parse_memcpy_info(const std::string &memcpy_command, size_t &add,
                         size_t &count);

  // GCScaler variant: no gpgpu_sim* parameter. size_num_used_instructions = 1.
  void get_next_threadblock_traces(
      std::vector<std::map<address_type, traced_instructions_by_pc> *>
          threadblock_traces,
      std::vector<std::vector<address_type> *> threadblock_traced_pcs,
      unsigned int gpu_device_id, unsigned int streamid, unsigned int kernelid,
      unsigned trace_version, unsigned int block_id_x, unsigned int block_id_y,
      unsigned int block_id_z, std::string kernel_name,
      traced_execution &static_trace_info);

  void kernel_finalizer(kernel_trace_t *trace_info);

  std::string get_extra_trace_info_filename() {
    return m_extra_trace_info_filename;
  }

 private:
  std::string kernellist_filename;
  std::string m_extra_trace_info_filename;
  bool m_is_extra_trace_enabled;
  dynamic_trace::Trace dyn_trace;
  std::string m_threadblocks_main_path;
  unsigned int m_kernel_id_filter_start;
  unsigned int m_kernel_id_filter_end;
};

int extractNumberAfterPattern(const std::string &str,
                               const std::string &pattern);
void parseKernelAndStreamID(const std::string &commandString, int &kernelID,
                             int &streamID, int &gpuDeviceID);

#endif
