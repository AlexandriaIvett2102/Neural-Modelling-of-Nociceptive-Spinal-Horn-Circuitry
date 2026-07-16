# Spiking Model of Spinal Dorsal Horn Pain Processing

A Python/NumPy spiking neural network model of the spinal dorsal horn
-- the first relay station in the central nervous system where
peripheral pain ("nociceptive") signals are processed before being
sent to the brain. This is a port and cleanup of an iterative series
of MATLAB prototypes (kept in `original_matlab/` for reference) into a
small, documented, tested Python package.

## The model

Three populations of [Izhikevich](https://www.izhikevich.org/publications/spikes.htm)
spiking neurons, connected according to distance in a simple 2D spatial
layout:

* **NS (nociceptor-specific) neurons** -- receive the incoming pain
  stimulus directly (a noisy current pulse partway through the
  simulation).
* **WDR (wide-dynamic-range) neurons** -- integrate input from other
  populations and are, in real dorsal horn physiology, thought to be
  the main encoders of perceived pain intensity.
* **INH (inhibitory interneurons)** -- provide feedback/feedforward
  inhibition, a simplified stand-in for local and descending
  inhibitory control of pain signaling.

Synaptic transmission uses delayed, exponentially-decaying currents
(fast excitatory / slower inhibitory kinetics, loosely AMPA/GABA-like)
between neurons connected via a distance-thresholded network. A
"top-down inhibition" term suppresses WDR activity in proportion to
its own recent drive.

The main experiment sweeps a **synaptic weight scaling factor**
(0.5x - 2.0x) across otherwise-identical networks and looks at how
population synchrony changes -- a simplified way of asking "what
happens to dorsal horn network activity as synaptic gain increases,"
which is one of the mechanisms proposed for central sensitization in
chronic pain.

## Parameter provenance

Not every number in this model is literature-derived -- some (distance
thresholds, delay ranges, stimulus amplitude, overall synaptic weight
magnitudes) were chosen to produce qualitatively interesting dynamics,
not fit to a specific dataset. But two parts of the model *are* taken
directly from published values, traced back through the MATLAB drafts:

* **Synaptic reversal potentials** (`E_rev_AMPA = E_rev_NMDA = 0` mV,
  `E_rev_GABA = -75` mV, hard-coded into the synaptic current terms in
  `simulate.py` as `E_EXC`/`E_INH`) are attributed in early drafts
  (`original_matlab/Draft_6.m`-`Draft_8.m`) to Humphries et al., 2009.
  The citation comment didn't survive into the later drafts, but the
  values it justified did, unchanged, all the way to `Draft_30.m` and
  into this port.
* **Izhikevich neuron-type parameters**: `a = 0.02, b = 0.2, c = -65,
  d = 8` for the NS/WDR populations are Izhikevich's own published
  "regular spiking" cortical excitatory neuron parameters (Izhikevich,
  2003); `d = 2` for the inhibitory population matches (half of) his
  "fast-spiking" interneuron parameters. The drafts don't cite this
  explicitly, but the numbers are the standard, recognizable constants
  from that model, not arbitrary choices -- each population then gets
  Gaussian jitter around those literature values for heterogeneity.

## Project layout

```
dorsal_horn_model/       Core package
    config.py             SimulationConfig -- all tunable parameters
    network.py             Neuron parameter draws + spatial connectivity
    simulate.py             The two simulation implementations (see below)
    analysis.py             Vectorized population-synchrony calculation
    plotting.py              Raster / synchrony / weight-sweep plots
run_experiment.py         Reproduces the full weight-factor sweep, saves figures/
validate_reference.py     Proves the fast implementation matches the slow one
notebooks/                Narrative walkthrough notebook
original_matlab/          The MATLAB source this was ported from
figures/                  Output of run_experiment.py
```

## Two implementations, on purpose

The original MATLAB computed every neuron's synaptic input with a
nested `for i / for j` loop over all neuron pairs, every time step.
That's how `simulate_reference_loop()` in `simulate.py` is written too
-- a direct, line-by-line translation, kept specifically so it can be
read side-by-side with `original_matlab/Draft_30.m` and trusted as a
"ground truth."

`simulate_network()` computes the *same* mathematics with NumPy array
broadcasting instead of Python loops, which is what actually runs the
full 240-neuron / 20,000-step experiment in about 20 seconds instead
of the hours the loop version would take. `validate_reference.py`
builds one shared small network, runs both implementations against it
with the same seeded RNG, and checks the resulting spike trains are
**bit-for-bit identical**:

```
$ python validate_reference.py
seed=0 weight_factor=1.5
  reference loop spikes : 27
  vectorized     spikes : 27
  spike trains identical: True
...
All checks passed: vectorized simulation matches the reference loop exactly.
```

## What changed vs. the original MATLAB (and why)

The MATLAB folder contains ~30 drafts of the same model. `Draft_30.m`
is the most complete, and is what this port is based on. Reading it
closely turned up a few genuine bugs, which are fixed here rather than
faithfully reproduced:

* **The weight-factor sweep didn't actually sweep anything.** MATLAB
  computed `g_AMPA`/`g_NMDA`/`g_GABA` scaled by `weight_factor`, but
  those matrices are never referenced in the simulation loop --
  the arrays that *are* used (`weights_exc`, `weights_inh`) were built
  with no `weight_factor` term at all. So each of the four subplots in
  the original figure just showed a different random network at
  identical effective synaptic strength. This port applies
  `weight_factor` directly to `weights_exc`/`weights_inh`
  (`network.py::build_connectivity`), so the sweep now does what it
  claims to -- see `figures/weight_sweep_raster_synchrony.png`, where
  higher weight factors visibly drive the network from sparse firing
  into strong, sustained synchronous bursting.
* **The synchrony-vs-time-argument order was swapped at the call
  site**, silently stretching the plotted time axis by 10x in some
  drafts. `analysis.py::compute_synchrony` takes explicit, named
  `dt` / `bin_width_ms` arguments instead of a positional pair that's
  easy to swap.
* **The final "synchrony across all variations" figure was an empty
  loop** -- it declared a title/legend but its `for` loop body was
  commented out, so nothing was ever plotted into it. This is now a
  real overlay plot (`plot_synchrony_overlay`, saved as
  `figures/synchrony_overlay.png`).
* **The per-step synchrony metric was extremely noisy** by
  construction (it could only take values that are multiples of
  `1/N_total` at `dt = 0.1` ms resolution). `compute_synchrony` bins
  over a configurable window (1 ms by default, matching what a couple
  of the MATLAB drafts were already reaching for) for a readable
  trace.
* **The connectivity block comments don't match the block indexing.**
  E.g. the comment on the first connectivity block says "connect NS to
  WDR neurons," but the row/column convention used everywhere else in
  the loop (`currents(i) += ... V(j)` under `connectivity(i, j)`)
  means that block actually drives NS *from* WDR. The numerical
  behavior is preserved as-is (since that's what produced the results
  being ported), but `network.py` documents the real wiring rather
  than the comment.

Everything else -- the Izhikevich parameters, the distance-based
connectivity, the exponential synaptic kernels, the top-down
inhibition term, the stimulus protocol -- is a direct, verified port.

## Running it

```bash
pip install -r requirements.txt

# Prove the fast and slow implementations agree (few seconds)
python validate_reference.py

# Run the full weight-factor sweep experiment (~1-2 minutes)
python run_experiment.py

# Or explore interactively
jupyter notebook notebooks/dorsal_horn_model_demo.ipynb
```

`run_experiment.py` accepts `--seed`, `--T`, and `--weight-factors` if
you want to try different durations or scaling factors.

## Using the package directly

```python
import numpy as np
from dorsal_horn_model import SimulationConfig, simulate_network, plot_raster, plot_synchrony

cfg = SimulationConfig(T=1000.0)
rng = np.random.default_rng(42)
result = simulate_network(cfg, weight_factor=1.5, rng=rng)

plot_raster(result)
plot_synchrony(result)
```

## Caveats

The neuron and synapse parameters draw on real, cited computational
neuroscience literature (see "Parameter provenance" above), but the
network as a whole was not fit or validated against real dorsal horn
recordings -- there's no dataset behind the connectivity, weight
magnitudes, or stimulus protocol; those were chosen to produce
qualitatively interesting dynamics. Treat this as a demonstration of
spiking-network modeling technique (heterogeneous, literature-grounded
Izhikevich populations, distance-based connectivity, delayed
conductance-style synapses, population synchrony analysis) built on
real building blocks, rather than as a validated model of nociception.

## References

* Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE
  Transactions on Neural Networks*, 14(6), 1569-1572.
* Humphries, M. D. et al. (2009). Cited in the original MATLAB drafts
  as the source for the AMPA/NMDA/GABA reversal potentials used here

  
