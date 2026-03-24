# ml_pipeline

This directory contains the local ML-side utilities and configs used with the LEP analysis inputs.

## Layout

- `EveNet-Full/`: upstream EveNet codebase and docs.
- `config/analysis.yaml`: sample list for plotting DataLoader parquet outputs.
- `util/plot_control_parquets.py`: simple data-vs-MC control plotting script for parquet files produced by `processor/DataLoader.py`.
- `util/build_evenet_input_from_parquet.py`: convert DataLoader parquet outputs into a simple EveNet-style `.npz` bundle plus metadata and an `event_info.yaml` skeleton.
- `util/evenet_parquet_common.py`: shared visible-tau and invisible-target helpers used by both scripts.

The ML utilities use awkward/vector `Momentum4D` objects for the reconstructed particle four-vectors and visible-tau sums, so the plotting and conversion paths share the same four-momentum handling.

## Parquet Plotting

The plotting script is intended for the filtered awkward parquet files written by the core analysis, for example:

- `filtered___tautau.parquet`
- `filtered___pion.parquet`
- `filtered___pipi.parquet`

It reads the `Samples` block from `config/analysis.yaml`, loads each parquet with `awkward`, and applies the same global MC normalization used in the core plotting code:

`norm_factor / initial_total_num_events * luminosity`

If no luminosity is available, it falls back to shape-only normalization.

If `Subcategories` is present in the config, the script further splits a sample by `event_category` before plotting. This is useful for breaking `Ztautau` into stacked subchannels while keeping the same overall normalization scheme. Any uncategorized remainder is added automatically as `<sample>_others`.

### Run

```bash
python3 /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/util/plot_control_parquets.py \
  --config /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/config/analysis.yaml \
  --output-dir /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/plots
```

### Current default plots

- `isolation_angle`
- `erad`
- `prad`
- `charged_e`
- `missing_pt`
- `nprong`
- `n_neutral`
- `thrust_neglog1m`
- one plot for each particle momentum feature: `Part_energy`, `Part_pt`, `Part_eta`, `Part_phi`
- one plot for each auxiliary particle feature under `Part_*`
- `tau_vis_prong_energy`
- `tau_vis_prong_pt`
- `tau_vis_prong_eta`
- `tau_vis_prong_phi`
- `tau_vis_prong_mass`
- `tau_vis_rho_energy`
- `tau_vis_rho_pt`
- `tau_vis_rho_eta`
- `tau_vis_rho_phi`
- `tau_vis_rho_mass`

If the parquet contains the truth branches written by `processor/DataLoader.py`, the script also writes MC-only truth plots:

- `truth_tau_pt`
- `truth_anti_tau_pt`
- `truth_tau_pair_pt`
- `truth_tau_pair_mass`
- `truth_nunu_pt`
- `target_invisible_energy`
- `target_invisible_pt`
- `target_invisible_eta`
- `target_invisible_phi`
- `target_invisible_mass`

## Config Format

`config/analysis.yaml` expects:

```yaml
Samples:
  sample_name:
    name: "label"
    is_data: true | false
    is_signal: true | false
    lumi: 46.3          # data only, optional
    norm_factor: 1458.9 # MC only, optional
    input_files:
      - "/path/to/file.parquet"
```

`input_files` may also use glob patterns.

Optional `Subcategories` format:

```yaml
Subcategories:
  Ztautau:
    Ztautau_pipi: [11]
    Ztautau_pirho: [12, 21]
    Ztautau_pilep: [13, 31, 14, 41]
```

The key under `Subcategories` should match the sample key or sample `name` from `Samples`.

## Parquet -> EveNet Input

`util/build_evenet_input_from_parquet.py` converts the awkward parquet files from `processor/DataLoader.py` into one combined EveNet-style `.npz` bundle.

It currently:

- converts `Part_*` into the point-cloud tensor `x`
- replaces Cartesian four-momentum with `energy`, `pt`, `eta`, `phi`
- builds event-level `conditions`
- applies the preselection `nprong == 2`
- assigns `classification` labels from sample name plus optional subcategory split
- builds two visible-tau assumptions for every event:
  - `tau_vis_prong`: all charged prong legs in each tau hemisphere
  - `tau_vis_rho`: prong legs plus nearby photons within `dR < 0.3`
- fills `x_invisible` from `truth_tau - tau_vis_target`, where `tau_vis_target` is chosen per tau from `event_category`
  - `Pi` and `Lepton` use the prong-only visible tau
  - `Rho` uses the prong-plus-photon visible tau
- writes monitoring plots under `<output-dir>/monitoring/`, including:
  - `nprong` before/after the preselection for each input sample
  - one histogram per global input field
  - one histogram per particle momentum feature: `Part_energy`, `Part_pt`, `Part_eta`, `Part_phi`
  - one histogram per auxiliary `Part_*` feature
  - `tau_vis_prong_energy`
  - `tau_vis_prong_pt`
  - `tau_vis_prong_eta`
  - `tau_vis_prong_phi`
  - `tau_vis_prong_mass`
  - `tau_vis_rho_energy`
  - `tau_vis_rho_pt`
  - `tau_vis_rho_eta`
  - `tau_vis_rho_phi`
  - `tau_vis_rho_mass`
  - `target_invisible_energy`
  - `target_invisible_pt`
  - `target_invisible_eta`
  - `target_invisible_phi`
  - `target_invisible_mass`
- uses the same data-vs-stacked-MC plotting style and normalization logic as `util/plot_control_parquets.py`
- writes `evenet_input.npz`, `evenet_input_metadata.json`, and `event_info.yaml`

The visible-tau and target-invisible definitions are shared with `util/plot_control_parquets.py`, so the monitor plots and the standalone plotting script use the same reconstruction assumptions.

Run:

```bash
python3 /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/util/build_evenet_input_from_parquet.py \
  --config /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/config/analysis.yaml \
  --output-dir /Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/evenet_inputs
```
