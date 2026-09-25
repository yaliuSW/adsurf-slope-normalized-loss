# Example data provenance

## Synthetic examples

The clean observations in `examples/data/` are direct forward solutions at the
specified frequencies and modal orders, calculated using QEDispInv and the
algorithm of Pan et al. (2026), https://doi.org/10.1785/0120250207.
The elastic relations match those used in inversion.

The gradient and crustal geometries originate from the ADsurf benchmarks.
The strong low-velocity-zone geometry is adapted from Yang et al. (2024),
https://doi.org/10.1007/s10712-024-09826-y.

Twenty noisy strong-LVZ examples include five realizations of each of four
1% global relative-RMS noise structures: independent Gaussian noise,
frequency-correlated noise, mode-wise bias and sparse outliers. The clean
observations, perturbations, true models and initial ensembles are retained in
the numerical arrays. True Vs is for evaluation and plotting, not optimization
or model selection.

## Mirandola field example

`field.npz` contains 161 active/passive dispersion picks digitized from vector
markers in Figure 15 of Yang et al. (2025), *Multi-thin-layered surface wave
dispersion curve inversion based on broad learning*,
https://doi.org/10.6038/cjg2024R0556. The borehole reference in
`borehole_reference.csv` was digitized from Figure 17 of that study.

Mirandola was investigated in the InterPACIFIC project; see Garofalo et al.
(2016), https://doi.org/10.1016/j.soildyn.2015.12.009. These are
publication-derived numerical observations and an external borehole reference,
not a newly acquired survey or a unique exact inversion target. No publisher
PDF or original published figure is redistributed.

The final example uses nine layers and 100 common, smooth initial Vs profiles.
Its seed and parameters are in `field.json`. The fundamental observations share
group 0; two higher groups are assigned ordered whole modal branches during
selection. The borehole is used only by the plotting code after selection.

## Numerical files

Each case has a JSON configuration and an NPZ array file. Internal units are
km, km/s, g/cm³ and Hz. `obs` columns are frequency, phase velocity and observed
group; model arrays are indexed by initial model and layer. Finite-layer
thicknesses are positive, with a final zero for the halfspace. Bound arrays and
the residual-normalization velocity grid are included.

Saved selections and the four main cases' optimization trajectories are in
`examples/results/`. They belong to the final method, with first-power slope
normalization and weights refreshed at every update. Both raw and retained
selections are kept, including non-improving and noisy examples. Licenses for
the numerical software are described in `THIRD_PARTY_NOTICES.md`.

`local_slope/` contains the fixed 500-model candidate experiment, its direct
forward observations, cached A/B curves and the double-precision metrics used
to illustrate the local slope correction. Missing physical RMSE entries denote
incomplete known-mode predictions; they are not filled or used in the scatter.
