"""Generate paper-style SI distribution figures.

This script intentionally leaves ``plot_si_improvement_comparison.py`` and the
original box/points diagnostic figure unchanged. It reuses the existing
evaluation helpers, then writes standalone figures that can be arranged later
in LaTeX.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import FIGURES_DIR
from .params import MPLCONFIG_DIR

MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot_si_improvement_comparison import (
    COLORS,
    METHODS,
    POSTER_GRID_STYLE,
    SI_VALUE_METHODS,
    evaluate_modes,
    plot_box_with_points,
    si_values_by_method,
)
from .test_3d_beam_map import summarize_results


OUTPUTS = {
    "ecdf": "si_distribution_ecdf_paper.png",
    "boxplot": "si_distribution_boxplot_paper.png",
    "violin": "si_distribution_violin_paper.png",
}
MEDIAN_ECDF_OUTPUT = "si_distribution_ecdf_median_poster.png"


PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
}


def _axis_limits(data):
    all_values = np.concatenate(data)
    return (
        np.floor(np.nanmin(all_values) / 10.0) * 10.0,
        np.ceil(np.nanmax(all_values) / 10.0) * 10.0,
    )


def _finish(fig, ax, output_path, dual_background=False):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()
    if dual_background:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        fig.savefig(output_path, dpi=300, transparent=True)
        white_output_path = output_path.with_name(f"{output_path.stem}_white_background{output_path.suffix}")
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.savefig(white_output_path, dpi=300, transparent=False, facecolor="white", edgecolor="white")
    else:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_ecdf(si_values, output_path):
    methods = list(SI_VALUE_METHODS)
    data = [si_values[method] for method in methods]
    x_min, x_max = _axis_limits(data)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        for method, values in zip(methods, data):
            sorted_values = np.sort(values)
            cdf = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
            ax.step(sorted_values, cdf, where="post", linewidth=1.9, color=COLORS[method], label=method)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Beam-level SI [dB]")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(loc="upper left", frameon=True, edgecolor="0.25")
        _finish(fig, ax, output_path)


def plot_median_ecdf(si_values, output_path, num_cases):
    methods = list(SI_VALUE_METHODS)
    data = [si_values[method] for method in methods]
    x_min, x_max = _axis_limits(data)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for method, values in zip(methods, data):
            sorted_values = np.sort(values)
            cdf = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
            median = float(np.median(values))
            q90 = float(np.percentile(values, 90.0))
            ax.step(
                sorted_values,
                cdf,
                where="post",
                linewidth=2.4,
                color=COLORS[method],
                label=method,
            )
            ax.vlines(median, 0.0, 0.5, color=COLORS[method], linestyle=":", linewidth=1.5)
            ax.plot(median, 0.5, marker="o", color=COLORS[method], markersize=6)
            median_offset = {
                "Ref SI": (8, -27),
                "RL": (-10, 10),
                "RL-init PSO": (8, 10),
                "PSO": (-30, -27),
            }[method]
            ax.annotate(
                f"{median:.1f} dB",
                xy=(median, 0.5),
                xytext=median_offset,
                textcoords="offset points",
                color=COLORS[method],
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": COLORS[method], "alpha": 0.9},
            )
            ax.vlines(q90, 0.0, 0.9, color=COLORS[method], linestyle="--", linewidth=1.6)
            ax.plot(q90, 0.9, marker="o", color=COLORS[method], markersize=6)
            q90_offset = {
                "Ref SI": (-28, -23),
                "RL": (-28, 9),
                "RL-init PSO": (-44, -23),
                "PSO": (-28, 9),
            }[method]
            ax.annotate(
                f"{q90:.1f} dB",
                xy=(q90, 0.9),
                xytext=q90_offset,
                textcoords="offset points",
                color=COLORS[method],
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": COLORS[method], "alpha": 0.9},
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.3)
        ax.axhline(0.9, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Beam-level SI [dB]", fontsize=16)
        ax.set_ylabel("Empirical CDF", fontsize=16)
        ax.set_title("Beam-Level SI ECDF", fontsize=17, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, **POSTER_GRID_STYLE)
        ax.legend(loc="upper left", facecolor="white", framealpha=1.0, edgecolor="0.25", fontsize=10)
        _finish(fig, ax, output_path, dual_background=True)


def plot_boxplot(si_values, output_path):
    methods = list(SI_VALUE_METHODS)
    data = [si_values[method] for method in methods]
    x_min, x_max = _axis_limits(data)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        box = ax.boxplot(
            data,
            vert=False,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            medianprops={"color": "black", "linewidth": 1.45},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.5},
            whiskerprops={"linewidth": 1.1, "color": "0.25"},
            capprops={"linewidth": 1.1, "color": "0.25"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "0.35",
                "markersize": 2.6,
                "alpha": 0.75,
            },
        )
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.23)
            patch.set_edgecolor(COLORS[method])
            patch.set_linewidth(1.35)

        ax.set_yticks(np.arange(1, len(methods) + 1))
        ax.set_yticklabels(methods)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Beam-level SI [dB]")
        ax.grid(True, axis="x", color="0.88", linewidth=0.7)
        _finish(fig, ax, output_path)


def plot_violin(si_values, output_path):
    methods = list(SI_VALUE_METHODS)
    data = [si_values[method] for method in methods]
    positions = np.arange(1, len(methods) + 1)
    x_min, x_max = _axis_limits(data)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        violins = ax.violinplot(
            data,
            positions=positions,
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, method in zip(violins["bodies"], methods):
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_alpha(0.28)
            body.set_linewidth(1.2)

        for pos, values, method in zip(positions, data, methods):
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.plot([q1, q3], [pos, pos], color=COLORS[method], linewidth=4.0, solid_capstyle="butt")
            ax.plot(med, pos, marker="o", color="black", markersize=4.2)

        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Beam-level SI [dB]")
        ax.grid(True, axis="x", color="0.88", linewidth=0.7)
        legend_handles = [
            Line2D([0], [0], color="0.4", linewidth=4, label="Interquartile range"),
            Line2D([0], [0], marker="o", color="black", linewidth=0, markersize=4.2, label="Median"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", frameon=True, edgecolor="0.25")
        _finish(fig, ax, output_path)


def run(seed_start, num_seeds, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = FIGURES_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _, results_by_method = evaluate_modes(seed_start, num_seeds)
    si_values = si_values_by_method(results_by_method)

    plot_box_with_points(
        si_values,
        FIGURES_DIR / "si_improvement_comparison_box_points.png",
        seed_start,
        num_seeds,
    )
    plot_ecdf(si_values, output_dir / OUTPUTS["ecdf"])
    plot_boxplot(si_values, output_dir / OUTPUTS["boxplot"])
    plot_violin(si_values, output_dir / OUTPUTS["violin"])

    print(f"Shared cases: seeds {seed_start}-{seed_start + num_seeds - 1} ({num_seeds} cases)")
    print("Method summary for poster table:")
    for method in METHODS:
        summary = summarize_results(results_by_method[method])
        print(
            f"{method:<12} | feasible {summary['feasible_count']}/{summary['count']} | "
            f"mean SI {summary['mean_si']:.2f} dB | "
            f"mean SI reduction {summary['mean_si_improvement']:.2f} +/- "
            f"{summary['std_si_improvement']:.2f} dB | "
            f"mean DL/UL loss {summary['mean_dl_gain_loss']:.2f}/"
            f"{summary['mean_ul_gain_loss']:.2f} dB | "
            f"runtime {summary['total_runtime_s']:.2f} s"
        )
    print("Saved paper-style SI distribution figures:")
    print(FIGURES_DIR / "si_improvement_comparison_box_points.png")
    for filename in OUTPUTS.values():
        print(output_dir / filename)


def run_median_ecdf(seed_start, num_seeds, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = FIGURES_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _, results_by_method = evaluate_modes(seed_start, num_seeds)
    si_values = si_values_by_method(results_by_method)
    output_path = output_dir / MEDIAN_ECDF_OUTPUT
    plot_median_ecdf(si_values, output_path, num_seeds)

    print(f"Shared cases: seeds {seed_start}-{seed_start + num_seeds - 1} ({num_seeds} cases)")
    print("Median summary for poster table:")
    for method in METHODS:
        results = results_by_method[method]
        median_best_si = float(np.median([item["best_si_db"] for item in results]))
        median_improvement = float(np.median([item["si_improvement_db"] for item in results]))
        print(
            f"{method:<12} | median SI {median_best_si:.2f} dB | "
            f"median SI reduction {median_improvement:.2f} dB"
        )
    print(f"Saved median-focused ECDF: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-style SI distribution figures.")
    parser.add_argument("--seed-start", type=int, default=5000, help="First seed in the evaluation range.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of shared theta-disabled cases to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory. Relative paths are written under the figures directory.",
    )
    parser.add_argument(
        "--median-only",
        action="store_true",
        help="Generate only the separate median-focused ECDF, leaving the existing ECDF file untouched.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.median_only:
        run_median_ecdf(args.seed_start, args.num_seeds, args.output_dir)
    else:
        run(args.seed_start, args.num_seeds, args.output_dir)
