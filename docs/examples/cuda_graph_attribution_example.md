# CUDA Graph Attribution Tracking

This guide shows how to use the CUDA graph attribution feature to understand the mapping between GPU kernels and CPU operations during runtime profiling.

## Problem

When profiling SGLang serving with `--profile`, decode phase uses CUDA graphs which "bake out" the CPU operations. This means:
- ✗ You only see `graph.replay()` in the profiler
- ✗ Individual kernels are visible but not linked to their originating CPU ops (layers, attention, MLP, etc.)
- ✗ Hard to understand what each kernel is doing

## Solution

The `--enable-cuda-graph-attribution` flag:
1. Records kernel → CPU op mapping during graph **capture** (at server init)
2. Exports the mapping to `cuda_graph_kernel_attribution.json`
3. **Automatically annotates** profiler traces with CPU op information
4. Adds profiler markers during graph **replay** for better trace readability

## Quick Start (3 Steps)

### Step 1: Launch server with attribution enabled

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --enable-cuda-graph-attribution
```

During initialization, you'll see:
```
INFO: CUDA graph kernel attribution map saved to cuda_graph_kernel_attribution.json (1234 unique kernels)
```

### Step 2: Run profiling with bench_serving

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 10 \
    --profile
```

**That's it!** The profiler will automatically:
- Find the attribution JSON file
- Annotate kernel events with their CPU operations
- Add attribution metadata to the trace

You'll see:
```
INFO: Annotated 1234 kernel events in trace with CUDA graph attribution
```

### Step 3: Analyze the results

**Option A: View in Chrome Trace Viewer**

Open the trace file in `chrome://tracing`. Kernel events will now show their CPU operations:
```
ampere_fp16_s1688gemm_fp16_128x128 [aten::linear]
fmha_cutlassF_f16_aligned_64x128 [aten::scaled_dot_product_attention]
```

**Option B: Generate a summary report**

```bash
python -m sglang.srt.utils.analyze_cuda_graph_profile \
    --trace /tmp/profile-TP-0-decode.trace.json.gz
```

Output:
```
============================================================================
CUDA Graph Profiling Report with Attribution
============================================================================

Server Configuration:
  Model: meta-llama/Llama-3.1-8B
  TP Size: 1
  Captured Batch Sizes: [1, 2, 4, 8, 16, 32]

Top CPU Operations by Total Time
----------------------------------------------------------------------------
CPU Operation                                      Total Time    Count   Avg Time
----------------------------------------------------------------------------
aten::linear                                        1234.56 ms    4567    270.23 us
aten::scaled_dot_product_attention                   890.12 ms    1234    721.45 us
aten::mul                                            123.45 ms    8901     13.87 us
...
```

## Detailed: What Gets Captured

The `cuda_graph_kernel_attribution.json` file contains:

```json
{
  "metadata": {
    "capture_bs": [1, 2, 4, 8, 16, 32],
    "num_tokens_per_bs": 1,
    "model": "meta-llama/Llama-3.1-8B",
    "tp_size": 1,
    "dp_size": 1
  },
  "kernel_to_cpu_op": {
    "ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn": [
      {
        "cpu_op": "aten::linear",
        "cuda_time_us": 245.3,
        "cpu_time_us": 12.1,
        "module": "LlamaForCausalLM/LlamaModel/layers.0/self_attn/q_proj",
        "stack_trace": [
          "llama.py:123",
          "modeling_utils.py:456"
        ]
      }
    ],
    "fmha_cutlassF_f16_aligned_64x128_rf_sm80": [
      {
        "cpu_op": "aten::scaled_dot_product_attention",
        "cuda_time_us": 312.7,
        "cpu_time_us": 8.4,
        "module": "LlamaForCausalLM/LlamaModel/layers.0/self_attn"
      }
    ]
  }
}
```

### Step 4: Correlate with runtime traces

When viewing Chrome traces from `--profile`, you can now:
1. Find `CUDA_GRAPH_REPLAY_BS{N}` markers in the trace
2. Look up kernel names in the attribution JSON
3. Understand which CPU operation each kernel came from

## Example Analysis Script

```python
import json

# Load attribution map
with open("cuda_graph_kernel_attribution.json") as f:
    attribution = json.load(f)

# Analyze a specific kernel
kernel_name = "ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn"
if kernel_name in attribution["kernel_to_cpu_op"]:
    for info in attribution["kernel_to_cpu_op"][kernel_name]:
        print(f"Kernel: {kernel_name}")
        print(f"  CPU Op: {info['cpu_op']}")
        print(f"  Module: {info.get('module', 'N/A')}")
        print(f"  CUDA Time: {info['cuda_time_us']:.2f} μs")
        if 'stack_trace' in info:
            print(f"  Stack: {' -> '.join(info['stack_trace'])}")

# Find all kernels for a specific layer
target_module = "layers.0/self_attn"
print(f"\nKernels in {target_module}:")
for kernel, infos in attribution["kernel_to_cpu_op"].items():
    for info in infos:
        if target_module in info.get("module", ""):
            print(f"  - {kernel}: {info['cpu_op']}")
```

## Performance Impact

- **Capture time**: Slightly slower (~5-10%) due to additional profiling with stack traces
- **Runtime**: Zero overhead when not actively profiling
- **Memory**: ~1-5 MB for attribution JSON file

## Compatibility

- Works with: `--enable-torch-compile`, `--tp-size`, `--dp-size`
- Not needed for: Prefill/extend phases (they already show full attribution)
- Best used with: `--enable-profile-cuda-graph` for complete capture-time profiling

## How It Works Behind the Scenes

When you enable `--enable-cuda-graph-attribution`:

1. **During server initialization** (capture phase):
   - Profiler runs with `with_stack=True` and `with_modules=True`
   - Captures full call stack and module hierarchy for each operation
   - Builds kernel → CPU op mapping from profiler events
   - Exports to `cuda_graph_kernel_attribution.json`

2. **During runtime profiling** (when you run `bench_serving --profile`):
   - Profiler automatically looks for `cuda_graph_kernel_attribution.json`
   - Annotates each kernel event in the trace with its CPU operation
   - Adds metadata to event args for detailed inspection
   - Saves annotated trace (no separate files needed!)

3. **When viewing traces**:
   - Kernel names include CPU operations in brackets
   - Event metadata includes full attribution details
   - Chrome trace viewer shows everything inline

## Troubleshooting

### No attribution file generated

- Check that `--enable-cuda-graph-attribution` is set
- Ensure CUDA graphs are enabled (not disabled with `--disable-cuda-graph`)
- Check logs for "CUDA graph kernel attribution map saved" message

### Empty or minimal attribution

- Some operations may not have meaningful names (generic kernels)
- Try with a larger model or more diverse operations
- Check that model actually uses CUDA graphs for decode (most models do by default)

### Attribution doesn't match runtime kernels

- Attribution is captured during graph creation with specific batch sizes
- Runtime may use different batch sizes (but kernels should be similar)
- Check that `capture_bs` in metadata matches your workload
