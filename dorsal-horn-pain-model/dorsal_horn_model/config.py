"""Simulation parameters for the dorsal horn network model.

Every field here has a direct counterpart in the original MATLAB script
(``original_matlab/Draft_30.m``); the docstring on each field notes the
MATLAB variable it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for one dorsal-horn network simulation.

    Attributes
    ----------
    N_NS:
        Number of nociceptor-specific (NS) neurons. MATLAB: ``N_NS``.
    N_WDR:
        Number of wide-dynamic-range (WDR) neurons. MATLAB: ``N_WDR``.
    N_INH:
        Number of inhibitory interneurons. MATLAB: ``N_INH``.
    dt:
        Integration time step, in milliseconds. MATLAB: ``dt``.
    T:
        Total simulated time, in milliseconds. MATLAB: ``T``.
    noise_amplitude:
        Standard deviation of the background current noise injected into
        every neuron at every time step. MATLAB: ``noise_amplitude``.
    base_inhibition:
        Base scaling factor for the top-down inhibition applied to WDR
        neurons. MATLAB: ``base_inhibition``.
    delay_range:
        ``(min, max)`` synaptic conduction delay, in milliseconds.
        MATLAB: ``delay_range``.
    stimulus_window:
        ``(start_step, end_step)`` (1-indexed, inclusive, matching the
        MATLAB ``nociceptor_input(200:400, :)`` slice) during which a
        noisy stimulus current is delivered to the NS population.
    stimulus_mean / stimulus_std:
        Mean and standard deviation of the stimulus current delivered to
        NS neurons during ``stimulus_window``.
    """

    N_NS: int = 40
    N_WDR: int = 160
    N_INH: int = 40
    dt: float = 0.1
    T: float = 2000.0
    noise_amplitude: float = 0.5
    base_inhibition: float = 0.2
    delay_range: Tuple[float, float] = (1.0, 5.0)
    stimulus_window: Tuple[int, int] = (200, 400)
    stimulus_mean: float = 100.0
    stimulus_std: float = 20.0

    @property
    def N_neurons(self) -> int:
        """NS + WDR (i.e. all non-inhibitory neurons). MATLAB: ``N_neurons``."""
        return self.N_NS + self.N_WDR

    @property
    def N_total(self) -> int:
        """Every neuron in the network (NS + WDR + INH)."""
        return self.N_neurons + self.N_INH

    @property
    def num_steps(self) -> int:
        """Number of simulation time steps. MATLAB: ``T/dt``."""
        return int(round(self.T / self.dt))
