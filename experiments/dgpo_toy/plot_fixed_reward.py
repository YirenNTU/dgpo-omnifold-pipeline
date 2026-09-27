"""Render the recorded fixed-reward trajectory without additional model evaluations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory: Path):
    report = json.loads((directory/"report.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, color, label in (("dgpo", "#2764a5", "DGPO"),
                              ("pathwise", "#cc6b23", "Pathwise control")):
        rows = [r for r in report["policy_histories"][arm] if "monitor_gain" in r]
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["monitor_gain"] for r in rows], color=color, label=label)
        axes[0].fill_between(steps, [r["monitor_lo95"] for r in rows],
                             [r["monitor_hi95"] for r in rows], color=color, alpha=.14)
        axes[1].plot(steps, [r["monitor_joint_signal"] for r in rows], color=color, label=label)
    axes[0].axhline(.1, color="#888888", linestyle=":", linewidth=1, label="Declared gain threshold")
    axes[0].set_ylabel("Mean frozen-reward gain from step 0")
    axes[1].set_ylabel("Known toy joint statistic")
    for ax in axes:
        ax.axhline(0., color="#aaaaaa", linewidth=.7)
        for step in sorted({v["step"] for v in report.get("resume_verification", {}).values()}):
            ax.axvline(step, color="#777777", linestyle="--", linewidth=.8)
        ax.set_xlabel("Policy update")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Fixed classifier, unchanged optimizer: training length only", fontsize=13)
    axes[0].set_title("Same independent monitor panel; bands = 95% evaluation CI", fontsize=9)
    axes[1].set_title("Secondary diagnostic, not distribution closure", fontsize=9)
    output = directory/"trajectory.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(plot(parser.parse_args().directory))
