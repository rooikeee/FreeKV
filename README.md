# FreeKV
FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

## Setup

### 1. Clone the repository
```bash
git clone --recursive https://github.com/sjtu-zhao-lab/FreeKV.git
cd FreeKV
```

### 2. Patch third-party dependencies

After cloning, apply the following patches to the submodules:

**flashinfer** – add GQA group sizes 5 and 7:
```bash
sed -i '/group_size == 4) {.*\\$/{
N;N;
s/\(__VA_ARGS__.*\\\)/\1\n  } else if (group_size == 5) {                              \\\n    constexpr size_t GROUP_SIZE = 5;                         \\\n    __VA_ARGS__                                              \\\n  } else if (group_size == 7) {                              \\\n    constexpr size_t GROUP_SIZE = 7;                         \\\n    __VA_ARGS__                                              \\/
}' 3rdparty/flashinfer/include/flashinfer/utils.cuh
```

**raft** – add bfloat16 support to vectorized IO:
```bash
# Add cuda_bf16.h include
sed -i '/#include <cuda_fp16.h>/a #include <cuda_bf16.h>' \
  3rdparty/raft/cpp/include/raft/util/vectorized.cuh

# Add bfloat16 IOType specializations
sed -i '/struct IOType<__half2, 4> {/{
N;N;
a\
\
template <>\
struct IOType<__nv_bfloat16, 1> {\
  // 1 x 16 bits = 16 bits. Use nv_float16 itself.\
  // Could also use __half, they are the same underlying type.\
  typedef __nv_bfloat16 Type;\
};\
template <>\
struct IOType<__nv_bfloat16, 2> {\
  // 2 x 16 bits = 32 bits. Use nv_half2 (alias for __half2).\
  typedef __nv_bfloat162 Type;\
};\
template <>\
struct IOType<__nv_bfloat16, 4> {\
  // 4 x 16 bits = 64 bits. Use uint2 (2 x 32 bits).\
  typedef uint2 Type;\
};\
template <>\
struct IOType<__nv_bfloat16, 8> {\
  // 8 x 16 bits = 128 bits. Use uint4 (4 x 32 bits).\
  typedef uint4 Type;\
};\
template <>\
struct IOType<__nv_bfloat162, 1> {\
  // 1 x (2 x 16 bits) = 32 bits. Use nv_half2 itself.\
  typedef __nv_bfloat162 Type;\
};\
\
template <>\
struct IOType<__nv_bfloat162, 2> {\
  // 2 x (2 x 16 bits) = 64 bits. Use uint2.\
  typedef uint2 Type;\
};\
\
template <>\
struct IOType<__nv_bfloat162, 4> {\
  // 4 x (2 x 16 bits) = 128 bits. Use uint4.\
  typedef uint4 Type;\
};\
\

}' 3rdparty/raft/cpp/include/raft/util/vectorized.cuh

# Ensure correct spacing before int32_t block
sed -i '/^struct IOType<__nv_bfloat162, 4>/,+3s/^};$/};\n/' \
  3rdparty/raft/cpp/include/raft/util/vectorized.cuh
```

### 3. Build
```bash
cd source
pip install -e .
```

## Acknowledgements

FreeKV is built upon [ArkVale](https://github.com/pku-liang/ArkVale) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer). We thank the developers of these projects for their excellent work.
