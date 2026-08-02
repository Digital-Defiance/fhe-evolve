#!/usr/bin/env python3
"""Generate publication figures for PAPER_ABSTRACT.md from results/*.json.

Writes PDF + PNG assets into paper/figures/. Re-run after a new evolution
run, then rebuild the paper with `npm run paper`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401 — registers 3d projection

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"

TIERS = (1024, 2048, 4096, 8192, 16384)
# Okabe–Ito (colourblind-safe), one hue per N tier.
COLORS = {
    1024: "#0072B2",
    2048: "#E69F00",
    4096: "#009E73",
    8192: "#CC79A7",
    16384: "#D55E00",
}
# Distinct dash patterns so coincident traces (e.g. identical noise maxima
# at N=8192 and N=16384) remain readable in greyscale / overlap.
LINESTYLES = {
    1024: "-",
    2048: "--",
    4096: "-.",
    8192: ":",
    16384: (0, (3, 1, 1, 1)),
}
LABELS = {n: f"$N={n}$" for n in TIERS}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"wrote {path.relative_to(REPO)}")
    plt.close(fig)


def _load_frontier() -> list[dict]:
    return json.loads((RESULTS / "pareto-frontier.json").read_text())


def _load_convergence() -> dict[int, list[dict]]:
    rows = json.loads((RESULTS / "convergence.json").read_text())
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[int(r["tier"])].append(r)
    for tier in by:
        by[tier].sort(key=lambda r: r["generation"])
    return by


def fig1_frontier_3d(points: list[dict]) -> None:
    """Latency × security × noise budget, coloured by N."""
    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    for n in TIERS:
        subset = [p for p in points if p["N"] == n]
        ax.scatter(
            [p["latencyMs"] for p in subset],
            [p["securityBits"] for p in subset],
            [p["noiseBudgetBits"] for p in subset],
            c=COLORS[n],
            s=8,
            alpha=0.55,
            depthshade=True,
            label=LABELS[n],
            linewidths=0,
        )

    ax.set_xlabel("Latency (ms)", labelpad=8)
    ax.set_ylabel("Estimated security (bits)", labelpad=8)
    ax.set_zlabel("Noise budget (bits)", labelpad=6)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", frameon=False, markerscale=1.6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Leave room on the right so the z-axis label is not clipped in PDF.
    fig.tight_layout()
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.05, top=0.95)
    _save(fig, "fig1-frontier-3d")


def fig2_latency_security(points: list[dict]) -> None:
    """Per-tier fronts on the latency–security plane."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    for n in TIERS:
        subset = [p for p in points if p["N"] == n]
        ax.scatter(
            [p["latencyMs"] for p in subset],
            [p["securityBits"] for p in subset],
            c=COLORS[n],
            s=12,
            alpha=0.65,
            label=LABELS[n],
            linewidths=0,
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Estimated security (estimator bits)")
    ax.set_xscale("log")
    ax.legend(loc="upper left", frameon=False, markerscale=1.4)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    _save(fig, "fig2-latency-security")


def fig3_convergence(by_tier: dict[int, list[dict]]) -> None:
    """Front size and best-per-objective traces per tier."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    panels = [
        (axes[0, 0], "paretoFrontSize", "Front size", False),
        (axes[0, 1], "bestLatencyMs", "Best latency (ms)", False),
        (axes[1, 0], "bestSecurityBits", "Best security (estimator bits)", False),
        (axes[1, 1], "bestNoiseBudgetBits", "Best noise budget (bits)", False),
    ]

    for ax, key, ylabel, _ in panels:
        for n in TIERS:
            rows = by_tier[n]
            gens = [r["generation"] for r in rows]
            vals = [r[key] for r in rows]
            ax.plot(
                gens,
                vals,
                color=COLORS[n],
                linestyle=LINESTYLES[n],
                linewidth=1.4,
                label=LABELS[n],
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        if key == "bestLatencyMs":
            ax.set_yscale("log")

    for ax in axes[1, :]:
        ax.set_xlabel("Generation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig3-convergence")


def main() -> None:
    _style()
    if not (RESULTS / "pareto-frontier.json").exists():
        raise SystemExit(f"missing {RESULTS / 'pareto-frontier.json'}")
    if not (RESULTS / "convergence.json").exists():
        raise SystemExit(f"missing {RESULTS / 'convergence.json'}")

    points = _load_frontier()
    print(f"frontier points: {len(points)}")
    fig1_frontier_3d(points)
    fig2_latency_security(points)
    fig3_convergence(_load_convergence())
    print(f"done → {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
