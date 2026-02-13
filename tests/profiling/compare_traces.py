#!/usr/bin/env python3
"""Compare autojac traces before and after optimization."""

import json
from pathlib import Path


def extract_duration(trace_file, function_name):
    """Extract the total duration of a function from a trace file.

    Args:
        trace_file: Path to the JSON trace file
        function_name: Name of the function to find (partial match)

    Returns:
        Duration in milliseconds, or None if not found
    """
    with open(trace_file) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    matching_events = [e for e in events if function_name in e.get("name", "")]

    if not matching_events:
        return None

    # Sum up all durations (in case there are multiple calls)
    total_dur_us = sum(e.get("dur", 0) for e in matching_events)
    # Convert microseconds to milliseconds
    return total_dur_us / 1000


def parse_filename(filename):
    """Parse model name and batch size from filename.

    Args:
        filename: e.g., "AlexNet()-bs4-cpu.json"

    Returns:
        tuple: (model_name, batch_size)
    """
    # Remove .json extension
    name = filename.replace(".json", "")
    # Split by -bs
    parts = name.split("-bs")
    model = parts[0].replace("()", "")
    batch_size = parts[1].split("-")[0]
    return model, batch_size


def compare_traces(old_dir, new_dir, device):
    """Compare traces for a specific device.

    Args:
        old_dir: Path to autojac_old directory
        new_dir: Path to autojac_new directory
        device: 'cpu' or 'cuda'

    Returns:
        list: List of result rows
    """
    results = []

    # Get all trace files for this device
    old_files = sorted(Path(old_dir).glob(f"*-{device}.json"))

    for old_file in old_files:
        new_file = Path(new_dir) / old_file.name

        if not new_file.exists():
            continue

        model, batch_size = parse_filename(old_file.name)

        # Extract durations for jac_to_grad
        jac_old = extract_duration(old_file, "jac_to_grad")
        jac_new = extract_duration(new_file, "jac_to_grad")

        # Extract durations for autojac_forward_backward
        fb_old = extract_duration(old_file, "autojac_forward_backward")
        fb_new = extract_duration(new_file, "autojac_forward_backward")

        # Calculate differences
        if jac_old and jac_new:
            jac_diff = jac_new - jac_old
            jac_pct = (jac_diff / jac_old) * 100 if jac_old else 0
        else:
            jac_diff = None
            jac_pct = None

        if fb_old and fb_new:
            fb_diff = fb_new - fb_old
            fb_pct = (fb_diff / fb_old) * 100 if fb_old else 0
        else:
            fb_diff = None
            fb_pct = None

        results.append(
            {
                "model": model,
                "batch_size": batch_size,
                "jac_old": jac_old,
                "jac_new": jac_new,
                "jac_diff": jac_diff,
                "jac_pct": jac_pct,
                "fb_old": fb_old,
                "fb_new": fb_new,
                "fb_diff": fb_diff,
                "fb_pct": fb_pct,
            },
        )

    return results


def format_ms(value):
    """Format a value as milliseconds."""
    if value is None:
        return "N/A"
    return f"{round(value)}"


def format_diff(diff, pct):
    """Format difference with percentage."""
    if diff is None or pct is None:
        return "N/A"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{round(diff)} ({sign}{round(pct)}%)"


def print_table(results, device):
    """Print comparison table for a device."""
    print(f"\n## {device.upper()} Results\n")

    # Print header
    header = "|Model|Batch Size|Time before (jac_to_grad)|Time after (jac_to_grad)|Difference (jac_to_grad)|Time before (autojac_forward_backward)|Time after (autojac_forward_backward)|Difference (autojac_forward_backward)|"
    separator = "|---|---|---|---|---|---|---|---|"

    print(header)
    print(separator)

    # Print rows
    for r in results:
        row = (
            f"|{r['model']}"
            f"|{r['batch_size']}"
            f"|{format_ms(r['jac_old'])} ms"
            f"|{format_ms(r['jac_new'])} ms"
            f"|{format_diff(r['jac_diff'], r['jac_pct'])}"
            f"|{format_ms(r['fb_old'])} ms"
            f"|{format_ms(r['fb_new'])} ms"
            f"|{format_diff(r['fb_diff'], r['fb_pct'])}|"
        )
        print(row)


def main():
    """Main entry point."""
    old_dir = Path("traces/autojac_old")
    new_dir = Path("traces/autojac_new")

    # Compare CPU traces
    cpu_results = compare_traces(old_dir, new_dir, "cpu")
    print_table(cpu_results, "cpu")

    # Compare CUDA traces
    cuda_results = compare_traces(old_dir, new_dir, "cuda")
    print_table(cuda_results, "cuda")


if __name__ == "__main__":
    main()
