# FreeKV
FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

## Setup

### 1. Clone the repository
```bash
git clone --recursive https://github.com/sjtu-zhao-lab/FreeKV.git
cd FreeKV
```

### 2. Create the environment with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install all dependencies:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **Note:** `flash-attn` requires building from source. Install it separately after the base dependencies:
> ```bash
> uv pip install flash-attn==2.6.3 --no-build-isolation
> ```

### 3. Patch third-party dependencies

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

### 4. Build
```bash
cd source
uv pip install -e .
```

## Usage

### Long Input (Gov Report Summarization)

**ArkVale** (baseline recall):
```bash
python source/pred.py --model qwen-2.5-chat-7b --dataset gov_report --max_gen 512 --data_idx 0 --recall_impl arkvale
```

**FreeKV** (speculative retrieval + correction):
```bash
python source/pred.py --model qwen-2.5-chat-7b --dataset gov_report --max_gen 512 --data_idx 0 --recall_impl cuda_cpy --spec_ret --corr 0.8
```

### Long Generation (LongGenBench)

**ArkVale**:
```bash
python source/pred.py --model qwen-2.5-chat-7b --dataset lgbench --max_gen 16384 --warmup 0 --data_idx 0 --recall_impl arkvale
```

**FreeKV**:
```bash
python source/pred.py --model qwen-2.5-chat-7b --dataset lgbench --max_gen 16384 --warmup 0 --data_idx 0 --recall_impl cuda_cpy --spec_ret --corr 0.9
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Model name (key in `config/model2path.json`) |
| `--dataset` | — | `gov_report`, `lgbench`, or `AIME24` |
| `--max_gen` | 8192 | Max new tokens to generate |
| `--budget` | 2048 | Total KV cache token budget (sink + recent + retrieved) |
| `--sink` | 512 | Number of sink tokens |
| `--recent` | 512 | Number of recent tokens |
| `--recall_impl` | `cuda_cpy` | Recall backend: `arkvale`, `torch_cpy`, or `cuda_cpy` |
| `--spec_ret` | off | Enable speculative retrieval |
| `--corr` | None | Correction cosine-similarity threshold (e.g. 0.8–0.9) |
| `--repeat_bsz` | 1 | Simulate batch size by repeating input |


## Acknowledgements

FreeKV is built upon [ArkVale](https://github.com/pku-liang/ArkVale) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer). We thank the developers of these projects for their excellent work.
