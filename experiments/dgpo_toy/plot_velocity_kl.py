"""Plot matched velocity/no-KL monitors, without selecting checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory):
    report = json.loads((directory/"report.json").read_text())
    fig, grid = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    axes = grid[0]
    for arm, color, label in (("no_kl", "#2764a5", "No KL (existing matched run)"),
                              ("velocity_kl", "#cc6b23", "Velocity MSE, coefficient 1")):
        rows = [row for row in report["policy_histories"][arm] if "monitor_gain" in row]
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, [row["monitor_gain"] for row in rows], color=color, label=label)
        axes[0].fill_between(steps, [row["monitor_lo95"] for row in rows],
                            [row["monitor_hi95"] for row in rows], color=color, alpha=.12)
        axes[1].plot(steps, [row["monitor_joint_signal"] for row in rows], color=color, label=label)
    truth = report["endpoints"]["initial"]["structure"]["truth_moment"][0]
    axes[1].axhline(truth, color="#777777", linestyle="--", label="Toy truth expectation")
    axes[0].set_ylabel("Frozen-reward gain from step 0")
    axes[1].set_ylabel("Mean context-corrected triple-phase cosine")
    axes[0].set_title("Matched fixed monitor; bands = 95% evaluation CI", fontsize=9)
    axes[1].set_title("Selected high-order statistic, not full closure", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Policy update")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0., .85))
    labels = ["Initial", "No KL", "Velocity MSE = 1"]
    names = ["initial", "no_kl", "velocity_kl"]
    colors = ["#999999", "#2764a5", "#cc6b23"]
    errors = [report["endpoints"][name]["structure"]["fourier_moment_rmse"] for name in names]
    variances = [report["endpoints"][name]["structure"]["marginal_variance_error_absmax"] for name in names]
    for ax, values, title, ylabel in (
            (grid[1, 0], errors, "Higher-order moments: closer is better", "RMSE to known truth moments"),
            (grid[1, 1], variances, "No-KL marginal distortion: lower is better", "Max absolute variance error (log scale)")):
        ax.bar(labels, values, color=colors, width=.55)
        for i, v in enumerate(values):
            ax.text(i, v*1.04, f"{v:.4g}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    grid[1, 1].set_yscale("log")
    grid[1, 1].set_ylim(.005, 1000)
    grid[1, 0].set_ylim(0, .6)
    fig.suptitle("10k matched DGPO: reward transfers without KL; velocity penalty 1 suppresses learning", fontsize=12)
    path = directory/"trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(plot(parser.parse_args().directory))
