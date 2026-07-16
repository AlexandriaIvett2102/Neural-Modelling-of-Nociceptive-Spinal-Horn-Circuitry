#!/usr/bin/env python3
"""Reproduce the synaptic-weight-factor sweep experiment from Draft_30.m.

Runs the full 240-neuron, 2000 ms dorsal-horn network at four synaptic
weight scaling factors, and saves the resulting raster/synchrony
figures under ``figures/``.

Usage
-----
    python run_experiment.py
    python run_experiment.py --seed 7 --T 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dorsal_horn_model import (
    SimulationConfig,
    plot_synchrony_overlay,
    plot_weight_sweep,
    simulate_network,
)

FIGURES_DIR = Path(__file__).parent / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--T", type=float, default=2000.0, help="simulation duration (ms)")
    parser.add_argument(
        "--weight-factors",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="synaptic weight scaling factors to sweep",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(T=args.T)
    FIGURES_DIR.mkdir(exist_ok=True)

    results = []
    t0 = time.time()
    for i, weight_factor in enumerate(args.weight_factors):
        rng = np.random.default_rng(args.seed + i)
        step_t0 = time.time()
        result = simulate_network(cfg, weight_factor, rng)
        n_spikes = int(result.spike_record.sum())
        print(
            f"  weight_factor={weight_factor:>4.1f}  "
            f"{n_spikes:>7d} spikes  "
            f"({time.time() - step_t0:5.1f}s)"
        )
        results.append(result)
    print(f"Total simulation time: {time.time() - t0:.1f}s")

    fig1 = plot_weight_sweep(results, args.weight_factors)
    fig1.savefig(FIGURES_DIR / "weight_sweep_raster_synchrony.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'weight_sweep_raster_synchrony.png'}")

    fig2 = plot_synchrony_overlay(results, args.weight_factors)
    fig2.savefig(FIGURES_DIR / "synchrony_overlay.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'synchrony_overlay.png'}")


if __name__ == "__main__":
    main()
