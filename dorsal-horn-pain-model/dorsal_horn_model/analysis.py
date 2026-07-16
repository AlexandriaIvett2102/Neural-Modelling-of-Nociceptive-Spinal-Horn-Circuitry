"""Population-synchrony analysis.

Replaces ``plotSynchronyFromSpikeTimes.m``, which computed, for every
single simulation time step, the fraction of neurons with a spike
within ``dt/2`` of that step by scanning every neuron's full spike
history (an O(N_total * num_steps) Python-unfriendly triple loop).

Two observations simplify this a lot:

1. Spike times only ever land exactly on simulation steps (multiples
   of ``dt``), so "any spike within ``dt/2`` of step *t*" is exactly
   "did this neuron spike at step *t*". The whole computation reduces
   to ``spike_record.mean(axis=1)``.
2. At ``dt = 0.1`` ms that per-step signal is extremely noisy (it can
   only take values that are multiples of ``1/N_total``). Binning into
   coarser windows (as the ``Draft_19``/``Draft_20`` annotated drafts
   attempted with a 1 ms ``timeResolution``) gives a much more
   readable synchrony trace, so that is offered here as the default.
"""

from __future__ import annotations

import numpy as np


def compute_synchrony(
    spike_record: np.ndarray, dt: float, bin_width_ms: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of neurons active in each time bin.

    Parameters
    ----------
    spike_record:
        Boolean array, shape ``(num_steps, N_total)``.
    dt:
        Simulation time step, in ms.
    bin_width_ms:
        Width of the synchrony analysis window, in ms. Must be >= dt.
        1 ms (i.e. 10 simulation steps at the model's default dt=0.1ms)
        matches the ``timeResolution`` used in the original annotated
        drafts.

    Returns
    -------
    bin_centers:
        Time (ms) at the center of each bin.
    synchrony:
        Fraction of neurons (0-1) that spiked at least once within
        each bin.
    """
    if bin_width_ms < dt:
        raise ValueError("bin_width_ms must be >= dt")

    steps_per_bin = max(1, round(bin_width_ms / dt))
    num_steps, n_neurons = spike_record.shape
    n_bins = num_steps // steps_per_bin
    if n_bins == 0:
        raise ValueError("Simulation is shorter than one synchrony bin")

    trimmed = spike_record[: n_bins * steps_per_bin]
    spikes_per_bin = trimmed.reshape(n_bins, steps_per_bin, n_neurons).any(axis=1)
    synchrony = spikes_per_bin.mean(axis=1)

    bin_centers = (np.arange(n_bins) + 0.5) * steps_per_bin * dt
    return bin_centers, synchrony
