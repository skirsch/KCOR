"""
Generate the v7.5 KCOR workflow figure for the preprint.

Output:
- documentation/preprint/figures/fig_kcor_workflow.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOX_FACE = "#fbfbfb"
BOX_EDGE = "#6b6b6b"
TEXT = "#222222"
ACCENT = "#4a4a4a"
OPTIONAL_EDGE = "#8a8a8a"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    dashed: bool = False,
    title_size: int = 11,
    body_size: int = 9,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        linewidth=2.0,
        edgecolor=OPTIONAL_EDGE if dashed else BOX_EDGE,
        facecolor=BOX_FACE,
        linestyle=(0, (5, 3)) if dashed else "solid",
    )
    ax.add_patch(box)

    ax.text(
        x + 0.015,
        y + h - 0.03,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
    )

    cursor_y = y + h - 0.065
    for line in lines:
        ax.text(
            x + 0.02,
            cursor_y,
            line,
            ha="left",
            va="top",
            fontsize=body_size,
            color=TEXT,
        )
        cursor_y -= 0.027


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, curved: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_length=7,head_width=5",
        mutation_scale=1,
        linewidth=1.8,
        color=ACCENT,
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(arrow)


def generate(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "(A)", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(0.35, 0.95, "(B)", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(0.83, 0.95, "(C)", fontsize=18, fontweight="bold", color=TEXT)

    # Panel A: observed data construction
    add_box(
        ax,
        0.05,
        0.73,
        0.22,
        0.16,
        "Fixed cohort input",
        [
            "• Enrollment frozen at baseline",
            "• Weekly deaths and risk sets",
            "• No post-enrollment reassignment",
        ],
    )
    add_box(
        ax,
        0.05,
        0.49,
        0.22,
        0.18,
        "Observed hazards and preprocessing",
        [
            r"• Compute $h_{\mathrm{obs},d}(t_{\mathrm{raw}})$",
            r"• Apply skip to get $h_d^{\mathrm{eff}}(t)$",
            r"• Optional $c_d(t)$ gives $h_d^{\mathrm{adj}}(t)$",
            r"• Accumulate $H_{\mathrm{obs},d}(t)$",
        ],
    )
    add_arrow(ax, (0.16, 0.73), (0.16, 0.67))

    # Panel B: core estimator
    add_box(
        ax,
        0.34,
        0.76,
        0.34,
        0.12,
        "1. Seed fit in nearest quiet window",
        [
            r"• Estimate $(\hat{k}_d,\hat{\theta}_{0,d}^{(0)})$",
            "• Anchor geometry near enrollment",
        ],
    )
    add_box(
        ax,
        0.34,
        0.59,
        0.34,
        0.12,
        r"2. Reconstruct $H_{0,d}^{\mathrm{eff}}(t)$",
        [
            "• Use all weeks after preprocessing",
            "• Quiet + wave periods together",
        ],
    )
    add_box(
        ax,
        0.34,
        0.42,
        0.34,
        0.12,
        r"3. Compute $\delta_{i,d}$ and $\Delta_d(t)$",
        [
            "• Persistent cumulative offsets",
            "• Align quiet windows after waves",
        ],
    )
    add_box(
        ax,
        0.34,
        0.25,
        0.34,
        0.12,
        "4. Pooled quiet-window refit",
        [
            r"• Refit $\hat{\theta}_{0,d}$ with $\hat{k}_d$ fixed",
            "• Use all aligned quiet windows",
        ],
    )
    add_box(
        ax,
        0.34,
        0.10,
        0.16,
        0.09,
        "Gamma-frailty inversion",
        [
            r"• Produce $\tilde H_{0,d}(t)$",
        ],
        title_size=10,
    )
    add_box(
        ax,
        0.52,
        0.10,
        0.16,
        0.09,
        r"$\mathrm{KCOR}(t)$ comparison",
        [
            "• Compare normalized cumulative hazards",
        ],
        title_size=10,
    )

    add_arrow(ax, (0.51, 0.76), (0.51, 0.71))
    add_arrow(ax, (0.51, 0.59), (0.51, 0.54))
    add_arrow(ax, (0.51, 0.42), (0.51, 0.37))
    add_arrow(ax, (0.43, 0.25), (0.43, 0.19))
    add_arrow(ax, (0.50, 0.145), (0.52, 0.145))

    add_arrow(ax, (0.66, 0.34), (0.66, 0.66), curved=1.25)
    ax.text(0.695, 0.50, "Iterate to convergence", fontsize=10, color=TEXT, rotation=90, va="center")

    # Panel C: optional NPH
    add_box(
        ax,
        0.75,
        0.52,
        0.20,
        0.22,
        "Optional epidemic-wave NPH module",
        [
            r"• Specify $c_d(t)$ independently",
            r"• Apply before $H_{\mathrm{obs},d}(t)$ accumulation",
            "• Outside the universal KCOR core",
        ],
        dashed=True,
        title_size=10,
    )
    add_arrow(ax, (0.75, 0.57), (0.27, 0.58), curved=0.22)

    ax.text(
        0.05,
        0.07,
        r"Workflow emphasis: frailty identification targets enrollment-time $\theta_{0,d}$ after rebasing,",
        fontsize=10,
        color=TEXT,
    )
    ax.text(
        0.05,
        0.04,
        "while depletion geometry is reconstructed over the full trajectory before final parameter estimation.",
        fontsize=10,
        color=TEXT,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="documentation/preprint/figures/fig_kcor_workflow.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    generate(Path(args.out))


if __name__ == "__main__":
    main()
