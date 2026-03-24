#!/usr/bin/env python3
"""
Analyze CUDA graph profiling traces with attribution.

Usage:
    python -m sglang.srt.utils.analyze_cuda_graph_profile \\
        --trace /tmp/profile-trace.json.gz \\
        --attribution cuda_graph_kernel_attribution.json

Or simply:
    python -m sglang.srt.utils.analyze_cuda_graph_profile \\
        --trace /tmp/profile-trace.json.gz
    (will auto-discover attribution file)
"""

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


def load_json(path):
    """Load JSON from regular or gzipped file."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def analyze_trace_with_attribution(trace_path, attribution_path=None):
    """Analyze profiling trace with CUDA graph attribution."""

    # Load trace
    print(f"Loading trace from {trace_path}...")
    trace = load_json(trace_path)

    # Auto-discover attribution file if not provided
    if attribution_path is None:
        trace_dir = os.path.dirname(trace_path) or '.'
        candidates = [
            os.path.join(trace_dir, "cuda_graph_kernel_attribution.json"),
            "cuda_graph_kernel_attribution.json",
            os.path.join(os.getcwd(), "cuda_graph_kernel_attribution.json"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                attribution_path = candidate
                break

    if attribution_path is None or not os.path.exists(attribution_path):
        print("ERROR: No attribution file found!")
        print("Please:")
        print("  1. Launch server with --enable-cuda-graph-attribution")
        print("  2. Specify attribution file with --attribution")
        return 1

    print(f"Loading attribution from {attribution_path}...")
    attribution_data = load_json(attribution_path)
    kernel_map = attribution_data.get("kernel_to_cpu_op", {})

    # Analyze trace events
    print("\nAnalyzing trace events...")
    kernel_times = defaultdict(lambda: {"total_time": 0.0, "count": 0, "cpu_ops": set()})

    for event in trace.get("traceEvents", []):
        if event.get("ph") == "X" and "dur" in event:  # Complete events
            kernel_name = event.get("name", "")
            duration_us = event["dur"]

            # Check if this kernel has attribution
            if kernel_name in kernel_map:
                for attr in kernel_map[kernel_name]:
                    cpu_op = attr["cpu_op"]
                    kernel_times[cpu_op]["total_time"] += duration_us
                    kernel_times[cpu_op]["count"] += 1
                    kernel_times[cpu_op]["cpu_ops"].add(cpu_op)

    # Generate report
    print("\n" + "=" * 80)
    print("CUDA Graph Profiling Report with Attribution")
    print("=" * 80)

    # Metadata
    metadata = attribution_data.get("metadata", {})
    print("\nServer Configuration:")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  TP Size: {metadata.get('tp_size', 'N/A')}")
    print(f"  Captured Batch Sizes: {metadata.get('capture_bs', 'N/A')}")
    print(f"  Tokens per BS: {metadata.get('num_tokens_per_bs', 'N/A')}")

    print(f"\nAttribution Coverage:")
    print(f"  Total unique kernels in map: {len(kernel_map)}")
    print(f"  CPU ops with runtime data: {len(kernel_times)}")

    # Top CPU operations by time
    if kernel_times:
        print("\n" + "-" * 80)
        print("Top CPU Operations by Total Time")
        print("-" * 80)
        sorted_ops = sorted(
            kernel_times.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )

        print(f"{'CPU Operation':<50} {'Total Time':>12} {'Count':>8} {'Avg Time':>12}")
        print("-" * 80)
        for cpu_op, stats in sorted_ops[:20]:
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            print(
                f"{cpu_op:<50} "
                f"{stats['total_time']/1000:>10.2f} ms "
                f"{stats['count']:>8} "
                f"{avg_time:>10.2f} us"
            )

        # Summary statistics
        total_time = sum(stats["total_time"] for stats in kernel_times.values())
        total_events = sum(stats["count"] for stats in kernel_times.values())
        print("-" * 80)
        print(f"{'TOTAL':<50} {total_time/1000:>10.2f} ms {total_events:>8}")

    # Per-layer breakdown (if module info available)
    print("\n" + "-" * 80)
    print("Per-Layer Breakdown")
    print("-" * 80)

    layer_times = defaultdict(lambda: {"total_time": 0.0, "count": 0})
    for event in trace.get("traceEvents", []):
        if event.get("ph") == "X" and "dur" in event:
            kernel_name = event.get("name", "")
            if kernel_name in kernel_map:
                for attr in kernel_map[kernel_name]:
                    module = attr.get("module", "")
                    if module:
                        # Extract layer number
                        if "layers." in module:
                            layer = module.split("layers.")[1].split("/")[0]
                            layer_key = f"layer {layer}"
                            layer_times[layer_key]["total_time"] += event["dur"]
                            layer_times[layer_key]["count"] += 1

    if layer_times:
        sorted_layers = sorted(
            layer_times.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        print(f"{'Layer':<20} {'Total Time':>15} {'Kernel Count':>15}")
        print("-" * 80)
        for layer, stats in sorted_layers[:10]:
            print(
                f"{layer:<20} "
                f"{stats['total_time']/1000:>13.2f} ms "
                f"{stats['count']:>15}"
            )
    else:
        print("No layer information available in attribution data.")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA graph profiling traces with attribution"
    )
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Path to Chrome trace file (*.json or *.json.gz)",
    )
    parser.add_argument(
        "--attribution",
        type=str,
        default=None,
        help="Path to cuda_graph_kernel_attribution.json (auto-discovered if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save report to file instead of printing to stdout",
    )

    args = parser.parse_args()

    if not os.path.exists(args.trace):
        print(f"ERROR: Trace file not found: {args.trace}")
        return 1

    return analyze_trace_with_attribution(args.trace, args.attribution)


if __name__ == "__main__":
    exit(main())
