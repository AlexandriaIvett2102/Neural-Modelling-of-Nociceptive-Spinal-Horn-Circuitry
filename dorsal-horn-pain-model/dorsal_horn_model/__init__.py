"""
dorsal_horn_model
==================

A small spiking neural network model of the spinal dorsal horn, the
first way-station for pain ("nociceptive") signals in the central
nervous system.

The network is built from three Izhikevich-neuron populations:

* **NS**  -- nociceptor-specific neurons that receive the incoming pain
  stimulus.
* **WDR** -- wide-dynamic-range neurons, which integrate nociceptive
  and non-nociceptive input and are thought to encode pain intensity.
* **INH** -- inhibitory interneurons that provide feedback/feedforward
  inhibition (a simplified stand-in for descending and local
  inhibitory control of pain).

This package is a Python/NumPy port of an iterative series of MATLAB
prototypes (see ``original_matlab/``). Two implementations of the
synaptic update are provided:

* :func:`dorsal_horn_model.simulate.simulate_reference_loop` -- a
  direct, line-by-line translation of the original nested ``for i /
  for j`` MATLAB loops. Kept for readability and as a correctness
  oracle; it is too slow to run at full scale.
* :func:`dorsal_horn_model.simulate.simulate_network` -- a vectorized
  NumPy implementation of the *same* mathematics, fast enough to run
  the full 240-neuron / 2000 ms experiment in a few seconds.

See ``validate_reference.py`` at the project root for a small-scale
check that the two agree.
"""

from .analysis import compute_synchrony
from .config import SimulationConfig
from .plotting import plot_raster, plot_synchrony, plot_synchrony_overlay, plot_weight_sweep
from .simulate import SimulationResult, simulate_network, simulate_reference_loop

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "simulate_network",
    "simulate_reference_loop",
    "compute_synchrony",
    "plot_raster",
    "plot_synchrony",
    "plot_synchrony_overlay",
    "plot_weight_sweep",
]
