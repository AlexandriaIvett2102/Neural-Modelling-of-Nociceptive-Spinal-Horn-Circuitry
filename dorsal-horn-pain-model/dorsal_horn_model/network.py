"""Neuron parameter and connectivity construction.

This mirrors the "Heterogeneous Izhikevich Model Parameters",
"Synaptic Parameters" and "Connectivity" sections of
``original_matlab/Draft_30.m``, factored out so both the reference
(nested-loop) and vectorized simulations build the network identically
from the same random draws.

Wiring convention
------------------
The MATLAB code (and this port) uses ``connectivity[i, j] == True`` to
mean "neuron *i* receives a synaptic contribution driven by neuron
*j*'s membrane potential/spikes" -- i.e. row = post-synaptic target,
column = pre-synaptic source. Note that in the original script the
in-code comments describing this block ("Connect nociceptor-specific
neurons to wide dynamic range neurons...") describe the *opposite*
direction from what the row/column indexing actually implements. We
preserve the original's literal numerical behaviour (since that is
what actually produced the results being ported) and document the
real wiring here rather than silently "fixing" the dynamics:

* rows ``[0, N_NS)``,               cols ``[N_NS, N_NS+N_WDR)``   -> WDR drives NS
* rows ``[N_NS, N_NS+N_WDR)``,      cols ``[N_total-N_INH, N_total)`` -> INH drives WDR
* rows ``[N_total-N_INH, N_total)``, cols ``[0, N_neurons)``      -> NS+WDR drives INH

See the README for more on this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class NetworkState:
    """Everything needed to run a simulation: neuron params + wiring."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    connectivity: np.ndarray  # bool, (N_total, N_total)
    weights_exc: np.ndarray  # float, (N_total, N_total)
    weights_inh: np.ndarray  # float, (N_total, N_total)
    delays: np.ndarray  # float, (N_total, N_total)
    nociceptor_input: np.ndarray  # float, (num_steps, N_NS)


def build_izhikevich_parameters(
    cfg: SimulationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw heterogeneous Izhikevich (a, b, c, d) parameters.

    Directly mirrors the "Heterogeneous Izhikevich Model Parameters"
    block of Draft_30.m: NS, WDR and INH populations get independently
    drawn ``a`` and ``d`` (their excitability/adaptation differs by
    neuron type), while ``b`` and ``c`` are drawn once across all
    neurons.

    The base values each population is jittered around are not
    arbitrary: ``a=0.02, b=0.2, c=-65, d=8`` are Izhikevich's own
    published "regular spiking" cortical excitatory neuron parameters
    (Izhikevich, 2003), used here for the NS/WDR populations. ``d=2``
    for the inhibitory population matches (half of) his "fast-spiking"
    interneuron parameters. Each neuron then gets Gaussian jitter
    around these literature values for heterogeneity.
    """
    a = np.concatenate(
        [
            0.02 + 0.005 * rng.standard_normal(cfg.N_NS),
            0.02 + 0.005 * rng.standard_normal(cfg.N_WDR),
            0.02 + 0.005 * rng.standard_normal(cfg.N_INH),
        ]
    )
    b = 0.2 + 0.01 * rng.standard_normal(cfg.N_total)
    c = -65.0 + 5.0 * rng.standard_normal(cfg.N_total)
    d = np.concatenate(
        [
            8.0 + 1.0 * rng.standard_normal(cfg.N_NS),
            8.0 + 1.0 * rng.standard_normal(cfg.N_WDR),
            2.0 + 0.5 * rng.standard_normal(cfg.N_INH),
        ]
    )
    return a, b, c, d


def build_connectivity(
    cfg: SimulationConfig, weight_factor: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build spatial connectivity and synaptic weights/delays.

    Neurons are scattered uniformly at random in a unit square and
    connected if they fall within a population-specific distance
    threshold, matching the "Connectivity" block of Draft_30.m.
    ``weight_factor`` scales the excitatory/inhibitory synaptic
    strengths (this is the parameter swept across in the portfolio
    experiment).

    Bug fix vs. the original script
    --------------------------------
    In Draft_30.m, ``weight_factor`` is only ever applied to
    ``g_AMPA`` / ``g_NMDA`` / ``g_GABA`` -- three conductance matrices
    that are computed but then never referenced anywhere in the
    simulation loop. The arrays that *are* used to compute synaptic
    currents, ``weights_exc`` and ``weights_inh``, are built from
    ``rand(...)`` with no ``weight_factor`` scaling at all. The result
    is that the original "synaptic weight sweep" experiment silently
    swept nothing -- each of its four subplots just showed a fresh
    random network at the same effective synaptic strength. This port
    fixes that by scaling ``weights_exc``/``weights_inh`` by
    ``weight_factor`` directly, so the sweep experiment actually shows
    what it claims to.
    """
    N_NS, N_WDR, N_INH = cfg.N_NS, cfg.N_WDR, cfg.N_INH
    N_neurons, N_total = cfg.N_neurons, cfg.N_total

    positions = rng.random((N_total, 2))
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))

    connectivity = np.zeros((N_total, N_total), dtype=bool)
    connectivity[0:N_NS, N_NS : N_NS + N_WDR] = (
        distances[0:N_NS, N_NS : N_NS + N_WDR] < 0.3
    )
    connectivity[N_NS : N_NS + N_WDR, N_total - N_INH : N_total] = (
        distances[N_NS : N_NS + N_WDR, N_total - N_INH : N_total] < 0.2
    )
    connectivity[N_total - N_INH : N_total, 0:N_neurons] = (
        distances[N_total - N_INH : N_total, 0:N_neurons] < 0.4
    )
    np.fill_diagonal(connectivity, False)

    weights_exc = rng.random((N_total, N_total)) * connectivity * 2.0 * weight_factor
    weights_inh = -rng.random((N_total, N_total)) * connectivity * weight_factor
    # Inhibitory interneurons project more strongly than the excitatory
    # populations do (MATLAB: weights_exc(end-N_INH+1:end, :) *= 2).
    weights_exc[N_total - N_INH : N_total, :] *= 2.0

    delay_lo, delay_hi = cfg.delay_range
    delays = (
        delay_lo + (delay_hi - delay_lo) * rng.random((N_total, N_total)) * connectivity
    )

    return connectivity, weights_exc, weights_inh, delays


def build_nociceptor_input(cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Noisy stimulus current delivered to the NS population.

    MATLAB: ``nociceptor_input(200:400, :) = 100 + 20*randn(201, N_NS)``.
    The stimulus window is 1-indexed and inclusive in the original;
    ``start - 1 : end`` below reproduces exactly that set of time steps
    with 0-indexed Python slicing.
    """
    nociceptor_input = np.zeros((cfg.num_steps, cfg.N_NS))
    start, end = cfg.stimulus_window
    n_steps = end - start + 1
    nociceptor_input[start - 1 : end, :] = cfg.stimulus_mean + cfg.stimulus_std * rng.standard_normal(
        (n_steps, cfg.N_NS)
    )
    return nociceptor_input


def build_network_state(
    cfg: SimulationConfig, weight_factor: float, rng: np.random.Generator
) -> NetworkState:
    """Draw one complete, randomly-parameterized network.

    Calling this once with a given ``rng`` and reusing the returned
    :class:`NetworkState` guarantees that
    :func:`dorsal_horn_model.simulate.simulate_network` and
    :func:`dorsal_horn_model.simulate.simulate_reference_loop` operate
    on an *identical* network, which is what makes them directly
    comparable (see ``validate_reference.py``).
    """
    a, b, c, d = build_izhikevich_parameters(cfg, rng)
    connectivity, weights_exc, weights_inh, delays = build_connectivity(
        cfg, weight_factor, rng
    )
    nociceptor_input = build_nociceptor_input(cfg, rng)
    return NetworkState(
        a=a,
        b=b,
        c=c,
        d=d,
        connectivity=connectivity,
        weights_exc=weights_exc,
        weights_inh=weights_inh,
        delays=delays,
        nociceptor_input=nociceptor_input,
    )
