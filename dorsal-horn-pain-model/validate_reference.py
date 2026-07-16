#!/usr/bin/env python3
"""Check that simulate_network (vectorized) agrees with the preserved
nested-loop reference translation of Draft_30.m's simulation loop.

Both implementations build their network from the same call sequence
against a freshly-seeded ``numpy.random.Generator`` (see
``dorsal_horn_model.network.build_network_state``), so given the same
seed they receive bit-identical neuron parameters, connectivity,
weights, delays, and stimulus. What differs is only *how* the O(N^2)
synaptic current sum is computed each step: Python ``for`` loops
(reference) vs. NumPy broadcasting (vectorized). This script builds one
shared network, runs the physics both ways with the noise RNG kept in
lockstep, and checks the resulting spike trains match exactly.

Run on a small network for a short duration -- the reference loop is
far too slow to run at the full experiment scale (that's the whole
reason :func:`dorsal_horn_model.simulate_network` exists).
"""

from __future__ import annotations

import math

import numpy as np

from dorsal_horn_model import SimulationConfig
from dorsal_horn_model.network import build_network_state
from dorsal_horn_model.simulate import E_EXC, E_INH, TAU_EXC, TAU_INH, SimulationResult, _simulate_vectorized


def _reference_from_state(
    cfg: SimulationConfig, state, weight_factor: float, rng: np.random.Generator
) -> SimulationResult:
    """Nested-loop physics (mirrors simulate_reference_loop) run against
    a pre-built NetworkState, so it can share that state exactly with
    the vectorized run below."""
    N_total = cfg.N_total
    N_NS, N_WDR = cfg.N_NS, cfg.N_WDR
    num_steps = cfg.num_steps
    dt = cfg.dt

    a, b, c, d = state.a, state.b, state.c, state.d
    connectivity = state.connectivity
    weights_exc = state.weights_exc
    weights_inh = state.weights_inh
    delays = state.delays
    nociceptor_input = state.nociceptor_input

    V = c.copy()
    u = np.zeros(N_total)
    spike_times: list[list[float]] = [[] for _ in range(N_total)]
    spike_record = np.zeros((num_steps, N_total), dtype=bool)

    for t in range(1, num_steps + 1):
        current_time = t * dt
        currents = cfg.noise_amplitude * rng.standard_normal(N_total)

        for i in range(N_total):
            total = 0.0
            for j in range(N_total):
                if not connectivity[i, j] or not spike_times[j]:
                    continue
                delayed_spike_time = spike_times[j][-1] + delays[i, j]
                if current_time > delayed_spike_time:
                    dt_since = current_time - delayed_spike_time
                    total += weights_exc[i, j] * math.exp(-dt_since / TAU_EXC) * (V[j] - E_EXC)
                    total += weights_inh[i, j] * math.exp(-dt_since / TAU_INH) * (V[j] - E_INH)
            currents[i] += total

        currents[:N_NS] += nociceptor_input[t - 1]
        wdr_mean = currents[N_NS : N_NS + N_WDR].mean()
        currents[N_NS : N_NS + N_WDR] -= (cfg.base_inhibition + 0.1 * wdr_mean) * wdr_mean

        dVdt = 0.04 * V**2 + 5.0 * V + 140.0 - u + currents
        dudt = a * (b * V - u)
        V = V + dt * dVdt
        u = u + dt * dudt

        spikes = V >= 30.0
        V = np.where(spikes, c, V)
        u = u + d * spikes

        for i in range(N_total):
            if spikes[i]:
                spike_times[i].append(current_time)
        spike_record[t - 1] = spikes

    return SimulationResult(cfg=cfg, weight_factor=weight_factor, spike_record=spike_record)


def run_comparison(cfg: SimulationConfig, weight_factor: float, seed: int) -> bool:
    # One shared network (parameters, connectivity, weights, delays,
    # stimulus) for both implementations.
    rng_build = np.random.default_rng(seed)
    state = build_network_state(cfg, weight_factor, rng_build)

    # Separate-but-identically-seeded RNGs for the per-step background
    # noise, so both runs draw the same noise sequence in the same order.
    rng_ref = np.random.default_rng(seed + 1)
    rng_vec = np.random.default_rng(seed + 1)

    ref_result = _reference_from_state(cfg, state, weight_factor, rng_ref)
    vec_result = _simulate_vectorized(cfg, state, weight_factor, rng_vec, record_voltage=False)

    match = np.array_equal(ref_result.spike_record, vec_result.spike_record)
    ref_spikes = int(ref_result.spike_record.sum())
    vec_spikes = int(vec_result.spike_record.sum())

    print(f"seed={seed} weight_factor={weight_factor}")
    print(f"  reference loop spikes : {ref_spikes}")
    print(f"  vectorized     spikes : {vec_spikes}")
    print(f"  spike trains identical: {match}")
    if not match:
        mismatch_steps = np.where((ref_result.spike_record != vec_result.spike_record).any(axis=1))[0]
        print(
            f"  first mismatching step: {mismatch_steps[0]} "
            f"(of {cfg.num_steps}), {len(mismatch_steps)} steps differ total"
        )
    return match


if __name__ == "__main__":
    small_cfg = SimulationConfig(
        N_NS=5,
        N_WDR=10,
        N_INH=5,
        dt=0.5,
        T=100.0,
        stimulus_window=(10, 20),
    )
    all_ok = True
    for seed in (0, 1, 2):
        all_ok &= run_comparison(small_cfg, weight_factor=1.5, seed=seed)

    if all_ok:
        print("\nAll checks passed: vectorized simulation matches the reference loop exactly.")
    else:
        raise SystemExit("Vectorized and reference implementations disagree -- see output above.")
