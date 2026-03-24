# ml_pipeline

This directory contains the local ML-side utilities and configs used with the LEP analysis inputs.

## Layout

- `EveNet-Full/`: upstream EveNet codebase and docs.
- `config/analysis.yaml`: sample list for plotting DataLoader parquet outputs.
- `util/plot_control_parquets.py`: simple data-vs-MC control plotting script for parquet files produced by `processor/DataLoader.py`.

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
