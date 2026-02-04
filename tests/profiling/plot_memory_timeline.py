"""
Script to plot memory timeline evolution from profiling traces.
Reads memory traces from json files and plots them on a single graph.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paths import TRACES_DIR


@dataclass
class MemoryFrame:
    timestamp: int
    total_allocated: int  # in bytes

    @staticmethod
    def from_event(event: dict):
        args = event["args"]
        return MemoryFrame(
            timestamp=event["ts"],
            total_allocated=args.get("Total Allocated"),
        )


def extract_memory_timeline(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)

    events = data["traceEvents"]
    print(f"Total events in trace: {len(events):,}")
    print("Extracting memory frames...")

    frames = [MemoryFrame.from_event(e) for e in events if e["name"] == "[memory]"]
    frames.sort(key=lambda frame: frame.timestamp)

    print(f"Found {len(frames):,} memory frames")

    timestamp_list = [frame.timestamp for frame in frames]
    total_allocated_list = [frame.total_allocated for frame in frames]

    return np.array([timestamp_list, total_allocated_list]).T


def plot_memory_timelines(experiment: str, folders: list[str]) -> None:
    timelines = list[np.ndarray]()
    for folder in folders:
        path = TRACES_DIR / folder / f"{experiment}.json"
        timelines.append(extract_memory_timeline(path))

    fig, ax = plt.subplots(figsize=(12, 6))
    for folder, timeline in zip(folders, timelines, strict=True):
        time = (timeline[:, 0] - timeline[0, 0]) // 1000  # Make time start at 0 and convert to ms.
        memory = timeline[:, 1]
        ax.plot(time, memory, label=folder, linewidth=1.5)

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Total Allocated (bytes)", fontsize=12)
    ax.set_title(f"Memory Timeline: {experiment}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    output_dir = Path(TRACES_DIR / "memory_timelines")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment}.png"
    print(f"\nSaving plot to: {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Plot saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Plot memory timeline from profiling traces.")
    parser.add_argument(
        "experiment",
        type=str,
        help="Name of the experiment under profiling (e.g., 'WithTransformerLarge()-bs4-cpu')",
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=str,
        help="Folder names containing the traces (e.g., autojac_old autojac_new)",
    )

    args = parser.parse_args()

    return plot_memory_timelines(args.experiment, args.folders)


if __name__ == "__main__":
    main()
