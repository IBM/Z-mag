#!/usr/bin/env python3
"""Build checkpoint-chain figures for theory_benchmarks.rst."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "checkpoint_chain"
OUT_CHAIN_PNG = SCRIPT_DIR / "checkpoint_chain_total_energy.png"
OUT_CHAIN_PDF = SCRIPT_DIR / "checkpoint_chain_total_energy.pdf"
OUT_DECOMP_PNG = SCRIPT_DIR / "checkpoint_chain_energy_decomposition.png"
OUT_DECOMP_PDF = SCRIPT_DIR / "checkpoint_chain_energy_decomposition.pdf"

CHECKPOINT_EVERY = 10

# High-contrast palette (colour-blind friendly)
COLOR_PH1 = "#2166ac"
RESTART_COLORS = ["#b2182b", "#d95f02", "#1b7837", "#7570b3"]
MARKER_SIZE = 5.0
LINE_WIDTH = 2.4
RESTART_LINE_WIDTH = 2.0


def _load(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def _plot_chain(ax, ph1, restarts):
    """Total-energy traces for Phase 1 and restart segments."""
    t_ph1 = ph1[:, 0]
    e_ph1 = ph1[:, 5]

    ax.plot(
        t_ph1,
        e_ph1,
        color=COLOR_PH1,
        lw=LINE_WIDTH + 0.4,
        label="Phase 1 (continuous reference)",
        zorder=2,
    )

    restart_times = []
    for i, (r, en) in enumerate(sorted(restarts.items())):
        color = RESTART_COLORS[i % len(RESTART_COLORS)]
        t_r = en[:, 0]
        e_r = en[:, 5]
        ax.plot(
            t_r,
            e_r,
            "-o",
            color=color,
            ms=MARKER_SIZE,
            mew=0.6,
            lw=RESTART_LINE_WIDTH,
            label=f"Restart {r}",
            zorder=3 + i,
        )
        restart_times.append(float(t_r[0]))
        ax.scatter(
            [t_r[0]],
            [e_r[0]],
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            zorder=6 + i,
        )

    # One legend entry for all restart-load times (avoids dotted/coloured clutter).
    if restart_times:
        ax.axvline(restart_times[0], color="0.35", ls="-", lw=1.0, alpha=0.55, zorder=1)
        for t0 in restart_times[1:]:
            ax.axvline(t0, color="0.35", ls="-", lw=1.0, alpha=0.55, zorder=1)
        ax.plot([], [], color="0.35", lw=1.0, label="Restart load time")

    # Labels R1–R4 near the start of each restart segment at the same y-level
    # so they are readable and easy to match to the corresponding trace.
    y_all_max = max(e_ph1.max(), max(en[:, 5].max() for en in restarts.values()))
    y_all_min = min(e_ph1.min(), min(en[:, 5].min() for en in restarts.values()))
    y_label = y_all_min + 0.92 * (y_all_max - y_all_min)
    x_label_offset = 0.015 * (t_ph1.max() - t_ph1.min())
    for i, (r, en) in enumerate(sorted(restarts.items())):
        color = RESTART_COLORS[i % len(RESTART_COLORS)]
        t0, e0 = float(en[0, 0]), float(en[0, 5])
        ax.text(
            t0 + x_label_offset,
            y_label,
            f"R{r}",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8, alpha=0.9),
        )
        # Small marker on the actual load point (helps visually connect label->trace).
        ax.scatter([t0], [e0], s=20, color=color, zorder=8)

    # Give a little headroom so labels and boxes are not clipped.
    ax.set_ylim(top=y_label + 0.03 * (y_all_max - y_all_min))

    ax.set_xlabel("Non-dimensional time")
    ax.set_ylabel("Total energy")
    ax.set_title("(a) Checkpoint chain — total energy")
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        fontsize=8.5,
        framealpha=0.95,
    )
    ax.grid(True, ls=":", alpha=0.35)


def _plot_decomposition(ax, ph1):
    """Show why the Phase 1 total energy can turn upward."""
    t = ph1[:, 0]
    ex = ph1[:, 2]
    an = ph1[:, 3]
    dem = ph1[:, 4]
    tot = ph1[:, 5]
    structure = ex + an

    ax.plot(t, dem, color="#4393c3", lw=LINE_WIDTH, label="Demag energy")
    ax.plot(t, structure, color="#d6604d", lw=LINE_WIDTH, label="Exchange + anisotropy")
    ax.plot(
        t,
        tot,
        color="0.15",
        lw=LINE_WIDTH + 0.2,
        ls="--",
        label="Total (= sum of active terms)",
    )

    ax.set_xlabel("Non-dimensional time")
    ax.set_ylabel("Energy")
    ax.set_title("(b) Phase 1 — energy decomposition")
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)
    ax.grid(True, ls=":", alpha=0.35)


def main() -> None:
    ph1 = _load(DATA_DIR / "energies_phase1.txt")
    restarts = {}
    for path in sorted(DATA_DIR.glob("energies_restart*.txt")):
        r = int(path.stem.replace("energies_restart", ""))
        restarts[r] = _load(path)

    # Two-panel figure for the benchmark page.
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=False)
    _plot_chain(axes[0], ph1, restarts)
    _plot_decomposition(axes[1], ph1)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(OUT_CHAIN_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(OUT_CHAIN_PDF, bbox_inches="tight")
    plt.close(fig)

    # Standalone decomposition (optional reuse).
    fig2, ax2 = plt.subplots(figsize=(8.6, 4.2))
    _plot_decomposition(ax2, ph1)
    fig2.tight_layout()
    fig2.savefig(OUT_DECOMP_PNG, dpi=240, bbox_inches="tight")
    fig2.savefig(OUT_DECOMP_PDF, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {OUT_CHAIN_PNG}")
    print(f"Wrote {OUT_DECOMP_PNG}")


if __name__ == "__main__":
    main()
