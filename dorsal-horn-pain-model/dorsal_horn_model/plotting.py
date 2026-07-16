"""Raster and synchrony plots.

Python/matplotlib equivalents of ``plotRasterFromSpikeTimes.m`` and
``plotSynchronyFromSpikeTimes.m``, with two deliberate improvements:

* The raster plot color-codes neurons by population (NS / WDR / INH)
  instead of plotting every spike as an identical black dot, which
  makes it possible to actually see how the stimulus propagates
  through the circuit.
* The synchrony plot uses :func:`dorsal_horn_model.analysis.compute_synchrony`,
  which is far less noisy than the raw per-time-step version in the
  original MATLAB helper (see that module's docstring for why).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .analysis import compute_synchrony
from .simulate import SimulationResult

POPULATION_COLORS = {
    "NS": "#d1495b",  # nociceptor-specific: red
    "WDR": "#1f78b4",  # wide-dynamic-range: blue
    "INH": "#2f9e44",  # inhibitory interneurons: green
}


def _population_bounds(result: SimulationResult) -> list[tuple[str, int, int]]:
    cfg = result.cfg
    return [
        ("NS", 0, cfg.N_NS),
        ("WDR", cfg.N_NS, cfg.N_NS + cfg.N_WDR),
        ("INH", cfg.N_NS + cfg.N_WDR, cfg.N_total),
    ]


def plot_raster(result: SimulationResult, ax: Axes | None = None, show_legend: bool = True) -> Axes:
    """Raster plot of spikes, color-coded by neuron population."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    spike_times = result.spike_times
    for name, lo, hi in _population_bounds(result):
        color = POPULATION_COLORS[name]
        for neuron_idx in range(lo, hi):
            times = spike_times[neuron_idx]
            if times.size:
                ax.plot(
                    times,
                    np.full(times.shape, neuron_idx),
                    ".",
                    color=color,
                    markersize=2,
                )

    if show_legend:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=name)
            for name, c in POPULATION_COLORS.items()
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(0, result.cfg.T)
    ax.set_ylim(-1, result.cfg.N_total)
    return ax


def plot_synchrony(
    result: SimulationResult, ax: Axes | None = None, bin_width_ms: float = 1.0
) -> Axes:
    """Population synchrony (fraction of neurons active per time bin)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    bin_centers, synchrony = compute_synchrony(result.spike_record, result.cfg.dt, bin_width_ms)
    ax.plot(bin_centers, synchrony * 100, linewidth=1.5, color="#333333")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Synchrony (%)")
    ax.set_xlim(0, result.cfg.T)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    return ax


def plot_weight_sweep(
    results: Sequence[SimulationResult],
    weight_factors: Sequence[float],
    bin_width_ms: float = 1.0,
) -> plt.Figure:
    """Reproduce Draft_30.m's main figure: one raster + synchrony row
    per synaptic-weight-factor variation."""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.2 * n), squeeze=False)

    for row, (result, wf) in enumerate(zip(results, weight_factors)):
        raster_ax, sync_ax = axes[row]
        plot_raster(result, ax=raster_ax, show_legend=(row == 0))
        raster_ax.set_title(f"Raster plot (weight factor = {wf:.1f})")

        plot_synchrony(result, ax=sync_ax, bin_width_ms=bin_width_ms)
        sync_ax.set_title(f"Synchrony (weight factor = {wf:.1f})")

    fig.tight_layout()
    return fig


def plot_synchrony_overlay(
    results: Sequence[SimulationResult],
    weight_factors: Sequence[float],
    bin_width_ms: float = 1.0,
) -> plt.Figure:
    """All weight-factor synchrony traces on one axis, for direct
    comparison (the original script's final figure declared this intent
    but never actually plotted anything into it -- an empty ``for``
    loop with no body -- so this is a genuine addition, not a port)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for idx, (result, wf) in enumerate(zip(results, weight_factors)):
        bin_centers, synchrony = compute_synchrony(result.spike_record, result.cfg.dt, bin_width_ms)
        color = cmap(idx / max(1, len(results) - 1))
        ax.plot(bin_centers, synchrony * 100, label=f"weight factor = {wf:.1f}", color=color)

    ax.set_title("Synchrony across synaptic weight variations")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Synchrony (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
