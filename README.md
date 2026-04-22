# ml_pipeline

This directory contains the local ML-side utilities and configs used with the LEP analysis inputs.

## Layout

- `EveNet-Full/`: upstream EveNet codebase and docs.
- `config/analysis.yaml`: sample list, configurable ML input feature lists, and normalization rules.
- `config/evenet_schema.yaml`: EveNet process topology and generation schema used to build the generated `event_info.yaml`.
- `config/preprocess_config.yaml`: static EveNet preprocessing wrapper that points to `config/generated_event_info.yaml`.
- `config/train.yaml`: repo-local EveNet training config following the upstream `share/finetune-example.yaml` pattern.
- `config/predict.yaml`: repo-local EveNet prediction config following the upstream `share/predict-example.yaml` pattern.
- `config/options.yaml`: local EveNet options overrides layered on top of `EveNet-Full/share/options/options.yaml`.
- `config/network.yaml`: local EveNet network overrides layered on top of `EveNet-Full/share/network/network-20M.yaml`.
- `util/plot_control_parquets.py`: simple data-vs-MC control plotting script for parquet files produced by `processor/DataLoader.py`.
- `util/build_evenet_input_from_parquet.py`: convert DataLoader parquet outputs into an EveNet `.npz` bundle plus metadata and a generated multi-process `event_info.yaml`. It also refreshes `config/generated_event_info.yaml` for EveNet preprocessing and training configs.
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
cd ml_pipeline
python3 util/plot_control_parquets.py \
  --config config/analysis.yaml \
  --output-dir plots
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
- one plot for each auxiliary particle feature listed in `Inputs.Part.Auxiliary`
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

Inputs:
  Part:
    Momentum: [energy, pt, eta, phi]
    Auxiliary:
      - Part_charge
      - Part_pdgId
      - ...
  Global:
    Fields:
      - Event_totalChargedEnergy
      - ...

Normalization:
  Sequential:
    Part_energy: log_normalize
    Part_pt: log_normalize
    Part_eta: normalize
    Part_phi: normalize_uniform
  Global:
    Event_totalChargedEnergy: log_normalize
    ...
  Invisible:
    energy: log_normalize
    pt: log_normalize
    eta: normalize
    phi: normalize_uniform
```

`input_files` may also use glob patterns.

`Inputs.Part.Auxiliary` and `Inputs.Global.Fields` control which features are:

- written into the EveNet point-cloud and condition tensors
- monitored by `util/build_evenet_input_from_parquet.py`
- plotted by `util/plot_control_parquets.py`

So if you want to add or remove `Part_*` or event-level inputs, edit `config/analysis.yaml` instead of the Python code.

`Normalization` is the source of truth for the tags written into the generated EveNet schema. Use it to control whether each feature is marked as:

- `none`
- `normalize`
- `log_normalize`
- `normalize_uniform`

These tags are copied into the generated `event_info.yaml` and then consumed by EveNet preprocessing for log scaling and metadata.

`config/evenet_schema.yaml` is the source of truth for the generated multi-process EveNet physics schema. The process names there must match the MC labels that will appear after subcategory expansion, because the converter uses it to build:

- `EVENT`
- `CLASSIFICATIONS`
- `CLASSLABEL`
- `GENERATIONS`

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

- keeps data samples in the monitoring plots, but excludes them from the final EveNet `.npz` payload
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
  - one histogram per global input field listed in `Inputs.Global.Fields`
  - one histogram per particle momentum feature: `Part_energy`, `Part_pt`, `Part_eta`, `Part_phi`
  - one histogram per auxiliary `Part_*` feature listed in `Inputs.Part.Auxiliary`
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
- writes `evenet_input.npz`, `evenet_input_metadata.json`, and `event_info.yaml` from MC samples
- refreshes `config/generated_event_info.yaml` from the same MC schema
- writes a real multi-process EveNet `event_info` with:
  - `INPUTS`
  - `EVENT`
  - `CLASSIFICATIONS`
  - `CLASSLABEL`
  - `GENERATIONS`
- writes `data.npz` and `data_metadata.json` from data samples in the same run

The visible-tau and target-invisible definitions are shared with `util/plot_control_parquets.py`, so the monitor plots and the standalone plotting script use the same reconstruction assumptions.

`classification` and the `CLASSLABEL` block in `event_info.yaml` are built from MC-only categories/subcategories. Data is treated as monitoring-only input and is never written into the EveNet training dataset.

The output split is:

- `evenet_input.npz`: MC-only payload for EveNet training / preprocessing
- `data.npz`: data-only payload written alongside it
- `event_info.yaml`: schema for the MC payload
- `config/generated_event_info.yaml`: latest generated MC schema for EveNet preprocessing

Run:

```bash
cd ml_pipeline
python3 util/build_evenet_input_from_parquet.py \
  --config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --output-dir evenet_inputs
```

## EveNet Preprocess

The converter above does **not** write the final EveNet training parquet shards by itself. It writes the inputs that EveNet preprocessing needs:

- `evenet_input.npz`
- `event_info.yaml`
- `config/generated_event_info.yaml`
- `config/preprocess_config.yaml`

To turn those into the final EveNet parquet files, run EveNet's own preprocessing step from [preprocess.py](EveNet-Full/preprocessing/preprocess.py). Use the static [config/preprocess_config.yaml](config/preprocess_config.yaml) for `--config`; it follows the same `default:` merge pattern as the upstream `share/*-example.yaml` files and points to:

- `config/generated_event_info.yaml`
- `config/resonance.yaml`

### Single NPZ -> train/val/test parquet

If you are using the single combined `.npz` produced by `util/build_evenet_input_from_parquet.py`, use EveNet's event-level split mode:

```bash
cd ml_pipeline
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --file evenet_inputs/evenet_input.npz \
  --split_ratio 0.8,0.1,0.1 \
  --store_dir evenet_inputs/preprocessed \
  -v
```

This writes:

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `shape_metadata.json`
- `normalization.pt`

The current repo copy of EveNet now validates the point-cloud feature count from `event_info` rather than assuming the old `(18, 7)` pretraining example shape, so custom `Part_*` feature layouts from `analysis.yaml` can pass preprocessing as long as the generated YAML and NPZ stay aligned.

### Explicit train/val/test NPZ splits

If you split your `.npz` files upstream, use EveNet's explicit split mode instead:

```bash
cd ml_pipeline
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --train /path/to/train.npz \
  --val /path/to/val.npz \
  --test /path/to/test.npz \
  --store_dir evenet_inputs/preprocessed \
  -v
```

## EveNet Train / Predict Configs

The repo-local EveNet configs under `config/` follow the same structure as the shipped `share` examples, but they point at the generated LEP schema:

- [config/train.yaml](config/train.yaml)
- [config/predict.yaml](config/predict.yaml)
- [config/options.yaml](config/options.yaml)
- [config/network.yaml](config/network.yaml)

Typical training flow:

```bash
cd ml_pipeline
shifter --image=docker:avencast1994/evenet:1.5 python3 EveNet-Full/scripts/train.py config/train.yaml
```

Typical prediction flow:

```bash
cd ml_pipeline
python3 EveNet-Full/scripts/predict.py config/predict.yaml
```

Before running either one, fill in the placeholder paths in `config/train.yaml`, `config/predict.yaml`, and `config/options.yaml`:

- `<data_parquet_dir>`
- `<normalization_file>`
- `<model_checkpoint_save_path>`
- `<log_save_dir>`
- `<ckpt_path>` for prediction

### What EveNet uses at training time

For EveNet training or prediction configs, point to the preprocessing output:

- `platform.data_parquet_dir`: directory containing `train.parquet`, `val.parquet`, `test.parquet`
- `options.Dataset.event_info`: the same `event_info.yaml`
- `options.Dataset.normalization_file`: `normalization.pt`

In practice, the end-to-end flow is:

1. Start from DataLoader parquet files.
2. Define or update the process topology / generation settings in `config/evenet_schema.yaml`.
3. Run [build_evenet_input_from_parquet.py](util/build_evenet_input_from_parquet.py) with `config/analysis.yaml` and `config/evenet_schema.yaml` to make `evenet_input.npz`, `event_info.yaml`, and refresh `config/generated_event_info.yaml`.
4. Run EveNet preprocessing with `config/preprocess_config.yaml` to make the final parquet shards plus normalization.
5. Fill in `config/train.yaml` or `config/predict.yaml` and run the corresponding EveNet script.

## Standalone Predict

For the LEP case study, the recommended inference path is the standalone predictor:

- [util/predict_evenet_from_raw_parquet.py](util/predict_evenet_from_raw_parquet.py)

It supports:

- `raw parquet` mode: start from analysis-side parquet files and re-run the `tautau` selection
- `converted parquet` mode: start from EveNet preprocessed parquet files such as `data.parquet` and `test.parquet`

For the current workflow, `converted parquet` mode is the main path.

### Converted parquet inference

Example:

```bash
cd ml_pipeline
python3 util/predict_evenet_from_raw_parquet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train.yaml \
  --checkpoint /path/to/last.ckpt \
  --output-dir /path/to/predict_output \
  --converted-parquet /path/to/data.parquet /path/to/test.parquet \
  --converted-split-fraction 0.5 \
  --batch-size 1024 \
  --num-gpus 4
```

Notes:

- `--num-gpus` is now event-chunk parallel for converted parquet mode, so a single large parquet can be split across multiple GPUs.
- `--converted-split-fraction` is used to rescale MC `evenet_weight` back to the full-sample normalization for data-vs-MC plots.
  - Example: if `test.parquet` corresponds to half of the original MC sample, pass `0.5`, so MC gets `x2` in the output plotting weight.
- Data is not assigned MC truth labels and keeps `evenet_weight = 1`.

### Prediction parquet contents

The converted-mode prediction output is intentionally self-contained for downstream evaluation and plotting. In addition to classification and neutrino prediction outputs, it stores:

- `evenet_pred_class_index`
- `evenet_pred_class_prob`
- `evenet_pred_class_name`
- `evenet_truth_class_index`
- `evenet_truth_class_name`
- `flags_valid`
- `event_weight`
- `evenet_weight`
- `pred_invisible_slot{0,1}_*`
- `target_invisible_slot{0,1}_*`
- `tau_vis_prong_slot{0,1}_energy`
- `tau_vis_prong_slot{0,1}_pt`
- `tau_vis_prong_slot{0,1}_eta`
- `tau_vis_prong_slot{0,1}_phi`
- `tau_vis_prong_slot{0,1}_valid`

This means the later evaluation step does **not** need the original converted parquet or `shape_metadata.json` anymore.

### Data-only converted parquet

If you only have `data.npz`, a practical way to turn it into a converted parquet for standalone prediction is:

```bash
cd ml_pipeline
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --train /path/to/data.npz \
  --test /path/to/data.npz \
  --store_dir /path/to/evenet_data
```

This is a conversion workaround for inference only. After that, use the produced `test.parquet` (or `train.parquet`) as the `--converted-parquet` input.

## Standalone Evaluation

The recommended summary plotting script is:

- [util/plot_evenet_prediction_summary.py](util/plot_evenet_prediction_summary.py)

It reads only the prediction parquet files produced by the standalone predictor. It no longer requires source converted parquet inputs.

Example:

```bash
cd ml_pipeline
python3 util/plot_evenet_prediction_summary.py \
  --analysis-config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --mc-parquet /path/to/test__evenet_pred.parquet \
  --data-parquet /path/to/data__evenet_pred.parquet \
  --output-dir /path/to/predict_summary
```

Current outputs include:

- `classification_confusion_weighted.png`
- `classification_confusion_row_normalized.png`
- `predicted_channel_purity.png`
- `predicted_class_data_vs_mc.png`
- `neutrino_truth_vs_pred_all.png`
- `neutrino_truth_vs_pred_kinematics_all.png`
- `neutrino_truth_vs_pred_<process>.png`
- `neutrino_truth_vs_pred_kinematics_<process>.png`
- `region_kinematics_<region>.png`
- `summary_metrics.yaml`

The summary plots use the stored prediction-parquet content directly:

- classification confusion is event-weighted with `evenet_weight`
- purity is shown as stacked truth-process yield per predicted channel, with `signal purity` written on the right-hand side
- neutrino comparison is split by correct classification vs mis-ID using different colors
- region plots compare data vs stacked MC and also overlay the MC truth reference

## Alignment Notes

When comparing standalone test-time prediction to the validation plots logged during training, keep these important differences in mind:

- Standalone neutrino inference is conditioned on the **predicted** class label.
- The diffusion monitoring during validation is typically conditioned on the dataset `classification`, i.e. the **truth** class label.
- The standalone predictor currently loads EMA weights when `EMA.enable: true` and `EMA.replace_model_after_load: true`.
- The standalone summary plots use `evenet_weight`, while training loss uses `apply_event_weight: false` in the current local config.

So a visible degradation from validation to standalone test does not automatically imply a broken prediction pipeline; some of the difference can come from these intentionally different evaluation conditions.
