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
3. Adds profiler markers during graph **replay** for better trace readability

## Usage

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

### Step 2: Run profiling

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 10 \
    --profile
```

### Step 3: Analyze the attribution

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

## Advanced: Integration with External Tools

The JSON format is designed for easy integration with profiling analysis tools:

```python
# Example: Annotate Chrome trace with attribution
import json

def annotate_chrome_trace(trace_file, attribution_file, output_file):
    with open(trace_file) as f:
        trace = json.load(f)

    with open(attribution_file) as f:
        attribution = json.load(f)

    # Add custom metadata to kernel events
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel" and "name" in event:
            kernel_name = event["name"]
            if kernel_name in attribution["kernel_to_cpu_op"]:
                # Add CPU op name as suffix
                cpu_ops = [info["cpu_op"] for info in attribution["kernel_to_cpu_op"][kernel_name]]
                event["name"] = f"{kernel_name} [{', '.join(set(cpu_ops))}]"

    with open(output_file, "w") as f:
        json.dump(trace, f)

# Usage
annotate_chrome_trace(
    "chrome_trace.json",
    "cuda_graph_kernel_attribution.json",
    "annotated_trace.json"
)
```

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
