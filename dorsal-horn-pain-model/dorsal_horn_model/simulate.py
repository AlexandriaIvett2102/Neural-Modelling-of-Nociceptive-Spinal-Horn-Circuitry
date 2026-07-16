"""Network simulation: Izhikevich dynamics + delayed synaptic currents.

Two implementations are provided that compute *the same thing*:

``simulate_reference_loop``
    A direct, nested-``for``-loop translation of the MATLAB simulation
    loop in Draft_30.m, preserved essentially line-for-line (see the
    comments cross-referencing the original). This is the easiest
    version to read against the MATLAB source, and it exists mainly to
    make it obvious that :func:`simulate_network` computes the same
    physics. It is O(N_total^2) per step *in Python*, so it is only
    practical for small networks / short durations.

``simulate_network``
    A NumPy-vectorized implementation of the identical mathematics,
    used for the actual portfolio-scale experiment (240 neurons,
    20,000 steps). Instead of looping over every ``(i, j)`` neuron
    pair in Python, it computes the whole ``(N_total, N_total)``
    synaptic contribution matrix in one shot per time step using
    broadcasting.

Both only look at each pre-synaptic neuron's *most recent* spike
(``spike_times[j][-1]`` in MATLAB), which is what makes the
vectorization possible: at each step we only need a single
"time since last spike" value per neuron, not the full spike history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .network import NetworkState, build_network_state

# Reversal potentials: AMPA/NMDA-like excitatory synapses reverse near
# 0 mV, GABA-like inhibitory synapses reverse near -75 mV. Early MATLAB
# drafts (Draft_6.m-Draft_8.m) attribute these specific values to
# Humphries et al., 2009; later drafts (including Draft_30.m, which
# this module is based on) hard-code the same numbers directly into
# the synaptic current expression but drop the citation comment.
E_EXC = 0.0
E_INH = -75.0
TAU_EXC = 2.0  # ms, excitatory synaptic decay time constant
TAU_INH = 5.0  # ms, inhibitory synaptic decay time constant


@dataclass
class SimulationResult:
    """Output of one network simulation run."""

    cfg: SimulationConfig
    weight_factor: float
    spike_record: np.ndarray  # bool, (num_steps, N_total)
    V_trace: np.ndarray | None = None  # optional, for debugging/plots

    @property
    def spike_times(self) -> list[np.ndarray]:
        """Per-neuron spike times in ms, as a list of arrays (like the
        MATLAB ``spike_times`` cell array)."""
        dt = self.cfg.dt
        steps = np.arange(1, self.spike_record.shape[0] + 1) * dt
        return [steps[self.spike_record[:, i]] for i in range(self.spike_record.shape[1])]


def simulate_network(
    cfg: SimulationConfig,
    weight_factor: float,
    rng: np.random.Generator,
    record_voltage: bool = False,
) -> SimulationResult:
    """Vectorized simulation -- use this one for real experiments.

    Parameters
    ----------
    cfg:
        Simulation configuration (network size, timing, noise, etc.).
    weight_factor:
        Scales excitatory/inhibitory synaptic strength (see
        :func:`dorsal_horn_model.network.build_connectivity`).
    rng:
        A ``numpy.random.Generator``. Pass the same seed used for
        :func:`simulate_reference_loop` to get a directly comparable
        network (see ``validate_reference.py``).
    record_voltage:
        If True, also store the full membrane-potential trace
        (``num_steps x N_total``). Off by default since it is the
        dominant memory cost for long runs.
    """
    state = build_network_state(cfg, weight_factor, rng)
    return _simulate_vectorized(cfg, state, weight_factor, rng, record_voltage)


def _simulate_vectorized(
    cfg: SimulationConfig,
    state: NetworkState,
    weight_factor: float,
    rng: np.random.Generator,
    record_voltage: bool,
) -> SimulationResult:
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
    last_spike = np.full(N_total, -np.inf)

    spike_record = np.zeros((num_steps, N_total), dtype=bool)
    V_trace = np.zeros((num_steps, N_total)) if record_voltage else None

    for step in range(num_steps):
        current_time = (step + 1) * dt  # matches MATLAB's `t*dt` with t = 1..num_steps

        # Background noise (MATLAB: noise_amplitude * randn(N_total, 1)).
        currents = cfg.noise_amplitude * rng.standard_normal(N_total)

        # --- Delayed synaptic currents, vectorized over all (i, j) pairs ---
        # For every ordered pair (i, j), if j is connected to i and j has
        # spiked before, its contribution to neuron i's current decays
        # exponentially with time since j's *last* spike, once that
        # delayed spike has "arrived" at i (current_time > delayed time).
        has_spiked = np.isfinite(last_spike)  # (N_total,)
        delayed_spike_time = last_spike[None, :] + delays  # (N_total, N_total), broadcast over rows
        time_since_arrival = current_time - delayed_spike_time
        active = connectivity & has_spiked[None, :] & (time_since_arrival > 0)

        V_pre = V[None, :]  # pre-synaptic membrane potential, broadcast over rows
        exc_contribution = weights_exc * np.exp(-time_since_arrival / TAU_EXC) * (V_pre - E_EXC)
        inh_contribution = weights_inh * np.exp(-time_since_arrival / TAU_INH) * (V_pre - E_INH)
        synaptic_current = np.where(active, exc_contribution + inh_contribution, 0.0)
        currents += synaptic_current.sum(axis=1)

        # Stimulus drive to the NS population.
        currents[:N_NS] += nociceptor_input[step]

        # Top-down inhibition of the WDR population, scaled by their own
        # mean drive (MATLAB: base_inhibition + 0.1*mean(...)) * mean(...)).
        wdr_slice = currents[N_NS : N_NS + N_WDR]
        wdr_mean = wdr_slice.mean()
        currents[N_NS : N_NS + N_WDR] -= (cfg.base_inhibition + 0.1 * wdr_mean) * wdr_mean

        # --- Izhikevich membrane dynamics (forward Euler) ---
        dVdt = 0.04 * V**2 + 5.0 * V + 140.0 - u + currents
        dudt = a * (b * V - u)
        V = V + dt * dVdt
        u = u + dt * dudt

        # --- Spike detection & reset ---
        spikes = V >= 30.0
        V = np.where(spikes, c, V)
        u = u + d * spikes
        last_spike = np.where(spikes, current_time, last_spike)

        spike_record[step] = spikes
        if V_trace is not None:
            V_trace[step] = V

    return SimulationResult(
        cfg=cfg, weight_factor=weight_factor, spike_record=spike_record, V_trace=V_trace
    )


def simulate_reference_loop(
    cfg: SimulationConfig,
    weight_factor: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Preserved, line-by-line nested-loop translation of Draft_30.m.

    This is intentionally *not* optimized: every ``(i, j)`` neuron pair
    is visited with an explicit Python ``for`` loop each time step,
    exactly like the original ``for i = 1:N ... for j = 1:N ...`` in
    MATLAB. It is kept so a reader can compare this function almost
    statement-for-statement against ``original_matlab/Draft_30.m`` and
    convince themselves :func:`simulate_network` computes the same
    thing (see ``validate_reference.py``, which runs both on a small
    network with the same seed and checks they agree).

    Only use this for small ``cfg`` (a handful of neurons, a few
    hundred ms) -- at the full 240-neuron / 2000 ms scale this is
    roughly 10,000x slower than :func:`simulate_network`.
    """
    state = build_network_state(cfg, weight_factor, rng)

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
    # MATLAB: spike_times = cell(N_total, 1);
    spike_times: list[list[float]] = [[] for _ in range(N_total)]
    spike_record = np.zeros((num_steps, N_total), dtype=bool)

    for t in range(1, num_steps + 1):  # MATLAB: for t = 1:T/dt
        current_time = t * dt

        currents = cfg.noise_amplitude * rng.standard_normal(N_total)

        # MATLAB: for i = 1:N_total / for j = 1:N_total
        for i in range(N_total):
            total = 0.0
            for j in range(N_total):
                if not connectivity[i, j]:
                    continue
                if not spike_times[j]:
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

    return SimulationResult(
        cfg=cfg, weight_factor=weight_factor, spike_record=spike_record, V_trace=None
    )
