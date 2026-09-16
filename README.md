# GCStack+ & GCScaler+

GCStack+ is a cycle accounting mechanism that builds a fine grained CPI stack for a GPU
kernel. GCScaler+ is an analytical interval model that rebuilds that CPI stack for
alternative GPU designs without re-running a cycle level simulation.

## Requirements

The tracer needs a real NVIDIA GPU and a matching driver. The models are pure CPU code and
run anywhere.

* CMake 3.28 or newer
* CUDA Toolkit, with `nvcc`, `cuobjdump` and `nvdisasm` on `PATH`
* GCC or Clang with C++17 support
* Protobuf (compiler and library), Boost (`iostreams` and `serialization`), OpenMP, zlib,
  bison, flex, liblzma, bzip2

On Debian or Ubuntu:

```bash
sudo apt install build-essential cmake bison flex \
    libprotobuf-dev protobuf-compiler \
    libboost-iostreams-dev libboost-serialization-dev \
    zlib1g-dev liblzma-dev libbz2-dev
```

NVBit is downloaded automatically by the build.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The examples below assume `GCPLUS` points at this repository.

```bash
export GCPLUS=$(pwd)
```

## 1. Trace a CUDA application

The tracer is an NVBit tool. You do not link it into the application. You inject it into an
unmodified binary through `CUDA_INJECTION64_PATH`.

```bash
mkdir -p ~/traces/vecadd && cd ~/traces/vecadd

CUDA_INJECTION64_PATH=$GCPLUS/build/lib/enhanced_tracer_tool.so /abs/path/to/vecadd args
```

**The trace lands in the current working directory.** The tracer writes a directory
named `traces_new` into the directory you launched from.

The result is:

```
traces_new/
  dynamic_trace.pb                     kernel launches, memcpys, grid and block dims
  threadblocks/device_0/stream_0/      one protobuf file per threadblock
    kernel_1/d_0_s_0_k_1_0,0,0.pb
  extra_info/
    enhanced_execution_info.json       control bits and register usage from the cubins
  stats.csv                            one line per kernel, instruction counts
```

## 2. Run GCStack+

GCStack+ is a trace driven cycle level simulator. It replays the trace on the baseline GPU
model and attributes every cycle to a stall category. Point `-trace` at the
`dynamic_trace.pb` file, not at the directory. Pass both configuration files.

```bash
cd ~/traces/vecadd

$GCPLUS/build/bin/gcstack_plus \
    -trace   $PWD/traces_new/dynamic_trace.pb \
    -config  $GCPLUS/gcstack/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM120_RTX5070/gpgpusim.config \
    -config  $GCPLUS/gcstack/gpu-simulator/configs/tested-cfgs/SM120_RTX5070/trace.config \
    > gpgpusim.log 2>&1
```

`gpgpusim.config` describes the microarchitecture and the memory hierarchy.
`trace.config` describes the instruction latencies and initiation intervals for the traced
ISA. Both live under `gcstack/gpu-simulator/`, and configurations for Turing, Ampere and
Blackwell parts are provided under `configs/tested-cfgs/`.

Outputs are written to the current working directory:

| File | Contents |
| --- | --- |
| `kernel_cpi_modern.bin` | per kernel CPI stack, the calibration target for GCScaler+ |
| `accelsim_cache_stat_modern.bin` | per access cache outcomes, the cache model input for GCScaler+ |
| `gpgpu_inst_stats.txt` | instruction mix per kernel |

## 3. Run GCScaler+

GCScaler+ takes the trace directory and the two files GCStack+ produced, and evaluates the
interval model. Work mode 4 is the mode used for the results in the paper. It searches for
the representative warp of each kernel that best reproduces the GCStack+ CPI stack, and then
runs the model with that warp.

```bash
cd ~/traces/vecadd

$GCPLUS/build/bin/gcscaler_plus \
    -C   $GCPLUS/gcscaler/configs/RTX5070.config \
    -t   $PWD/traces_new \
    -c   $PWD/accelsim_cache_stat_modern.bin \
    -k   $PWD/kernel_cpi_modern.bin \
    --csv $PWD/gcscaler.csv
```

`-t` takes the trace **directory**, unlike GCStack+. The format is detected automatically,
and the run confirms it with `[GCScaler] Detected protobuf trace format.`

The CSV holds one row per kernel, with the cycle count and the CPI broken into components:

```
Kernel,numThreadInst,numCycle,cpi,base,comData,comStruct,memData,memStruct,idle
0,311296,5682.89,0.0182556,0.000171327,3.49032e-05,0,0.00400018,0.000159976,0.0138892
```

## Notes

This work is based on the following works:

```bibtex
@inproceedings{huerta2025dissecting,
  title={Dissecting and modeling the architecture of modern gpu cores},
  author={Huerta, Rodrigo and Shoushtary, Mojtaba Abaie and Cruz, Jos{\'e}-Lorenzo and Gonzalez, Antonio},
  booktitle={Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture},
  pages={369--384},
  year={2025}
}

@inproceedings{cha2025gcstack_gcscaler,
  author    = {Hanna Cha and Sungchul Lee and Jounghoo Lee and Yeonan Ha and Joonsung Kim and Youngsok Kim},
  title     = {{GCStack+GCScaler: Fast and Accurate GPU Performance Analyses Using Fine-Grained Stall Cycle Accounting and Interval Analysis}},
  booktitle = {Proc. 52nd IEEE/ACM International Symposium on Computer Architecture (ISCA)},
  year      = {2025},
}

@article{cha2024gcstack,
  author  = {Hanna Cha and Sungchul Lee and Yeonan Ha and Hanhwi Jang and Joonsung Kim and Youngsok Kim},
  title   = {{GCStack: A GPU Cycle Accounting Mechanism for Providing Accurate Insight Into GPU Performance}},
  journal = {IEEE Computer Architecture Letters (CAL)},
  year    = {2024},
}
```
