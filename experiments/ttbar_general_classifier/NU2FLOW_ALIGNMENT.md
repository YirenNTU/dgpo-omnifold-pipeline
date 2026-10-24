# Frozen ttbar mixture evaluation: nu2flow alignment

The primary native comparison calls the existing EveNet-private functions,
without modifying that repository:

- `downstreams/nu2flow/distributions.py::preprocess`
- `downstreams/nu2flow/unfolding/unfold.py::main`
- `downstreams/plotting/unfolding.py::plot_uncertainty_with_ratio`

The native script reconstructs measured tops from reconstructed visible objects
plus neutrinos. Its unfolding truth axis is **parton MC top/lepton**, not the
reconstructed-visible-plus-MC-neutrino oracle target. The latter is the
truth-neutrino reconstruction control used as the uncertainty denominator.

| Item | Native nu2flow path | Separate all-candidate diagnostic |
| --- | --- | --- |
| Joint bins | Original mtt × observable, 4×4 | Same edges, 4×4 |
| Solver | Original RooUnfoldSvd, kreg=7, Hunfold(2) | RooUnfoldSvd, kreg=7 |
| Main uncertainty | Count error / truth-neutrino unfolded count error, per bin | Shape covariance RSS and count error / Poisson reference |
| Prediction per event | One categorical draw from K8, shared seed42 across arms | All K8 with event-normalized fractional weights |
| Covariance | Original histogram covariance, default no response-MC systematics | Full event contribution covariance, separate paired response bootstrap |
| Response/measurement | Same event pool, exactly as original main | Separate predeclared event partitions |
| Dilepton Δφ | Original signed Δφ/π, original [0,1] bins and misses | Explicit absolute Δφ/π |
| Closure | Same-sample residual; not independent coverage | Independent reference, nominal and tilted spectra |

Neither count errors nor shape errors from these two protocols should be
combined or labelled as identical. Native response-MC errors are not zero:
they are **not included**. The weighted diagnostic estimates them separately.
The native plot's lower panel compares to raw diffusion, not the old external
nu2flow model's paper numbers. No external model checkpoint is evaluated here.

Population limitation: the existing candidate cache keeps
`compute_base_recon_cut AND both assignments-mask` for physics and excludes
classifier-selection events. This differs from the notebook's
`num_bjet>=2, assigned b pT>25, assigned lepton pT>15` selection. The downloaded
candidate cache does not contain the full observed point cloud or num_bjet
needed to reproduce the original selection. Use the same existing population
for all arms and disclose this distinction; do not call this a full reproduction
of the historical analysis. The signed-Δφ restriction also rejects negative
values in the original script; do not silently replace it with absolute Δφ.

Run after the validation cache and original frozen head are available locally:

```bash
python experiments/ttbar_general_classifier/evaluate_native_nu2flow.py \
  config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5_batch1024.yaml \
  --output /absolute/path/to/downloaded/run \
  --evenet-root /Users/yirenwu/EveNet-private
```

This writes `native_nu2flow/`, including the exact input CSVs, original output
CSVs, source hashes, candidate indices, and bin-validity counts. Source code,
training, normalization, and existing results are not overwritten. Candidate
selection adds one realization of sampling noise, so small differences are not
proof of a robust gain. It requires no new diffusion generation or training.
