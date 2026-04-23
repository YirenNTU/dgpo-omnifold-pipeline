# LEP EveNet ml_pipeline

This directory contains the ML-side EveNet workflow for the LEP tau-pair analysis.

Project rules:

- Do not modify the central analysis or unfolding framework for EveNet-specific logic.
- Keep EveNet input building, training, prediction, diagnostics, and adapters inside `ml_pipeline`.
- When integrating with central QI or unfolding, export EveNet predictions into the central parquet schema.
- The default working directory for commands in this document is `ml_pipeline`.

## Directory Roles

- `config/analysis.yaml`: Main ML pipeline configuration. It defines sample input and raw parquet paths, feature lists, normalization rules, and optional prediction/export defaults.
- `config/evenet_schema.yaml`: EveNet process topology, classification labels, and truth-generation schema.
- `config/preprocess_config.yaml`: EveNet preprocessing wrapper that points to the generated event info.
- `config/train.yaml`: EveNet scratch training config.
- `config/train_pretrain.yaml`: EveNet pretrain finetuning config.
- `util/build_evenet_input_from_parquet.py`: Converts central/DataLoader parquet files into EveNet `.npz` input and monitoring plots.
- `util/predict_evenet_from_raw_parquet.py`: Standalone EveNet inference. The filename is legacy; the current script only consumes EveNet converted parquet files.
- `util/plot_evenet_prediction_summary.py`: Standalone EveNet prediction summary plots.
- `util/export_evenet_prediction_to_qi.py`: Converts EveNet prediction parquet files back into the central/QI parquet schema.
- `util/plot_qi_method_comparison.py`: Compares multiple central-schema methods such as Baseline, EveNet-Pretrain, and EveNet-Scratch.
- `util/evenet_parquet_common.py`: Shared visible tau, truth invisible target, and source-event matching utilities.

## Workflow Overview

```text
central tree_ana
  -> baseline/{sample}/filtered___raw.parquet
  -> baseline/{sample}/filtered___baseline.parquet

ml_pipeline build_evenet_input
  -> evenet_input.npz
  -> generated_event_info.yaml
  -> monitoring plots

EveNet preprocess
  -> train.parquet / val.parquet / test.parquet / data.parquet
  -> normalization.pt / shape_metadata.json

EveNet train + predict
  -> *__evenet_pred.parquet

export_evenet_prediction_to_qi
  -> qi_exports/{method}/{sample}/filtered___raw.parquet
  -> qi_exports/{method}/{sample}/filtered___{region}.parquet

central/QI evaluation or ml_pipeline summary plots
  -> Baseline vs EveNet-Pretrain vs EveNet-Scratch comparison
```

## 0. Produce Central Baseline Parquet

Run the central framework from the repository root. This step should not add EveNet-specific logic to the central code.

```bash
cd /path/to/lep_tree_ana
python3 bin/tree_ana \
  --config-yaml config/config.yaml \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline
```

Then point `ml_pipeline/config/analysis.yaml` at the central outputs:

```yaml
Samples:
  Ztautau:
    input_files:
      - /pscratch/.../baseline/Ztautau/filtered___baseline.parquet
    raw_files:
      - /pscratch/.../baseline/Ztautau/filtered___raw.parquet
```

`input_files` are the selected event universe used for EveNet input conversion. `raw_files` are the full raw event universe used when exporting back to central/QI/unfolding. Events without EveNet predictions are kept in the exported raw tree with invalid default reconstruction fields.

## 1. Build EveNet Input

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/build_evenet_input_from_parquet.py \
  --config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset
```

Main outputs:

- `/pscratch/.../dataset/evenet_input.npz`
- `/pscratch/.../dataset/evenet_input_metadata.json`
- `config/generated_event_info.yaml`
- Monitoring plots under the output directory

Current target invisible definition:

- Selected visible tau is prong plus photons within `dR < 0.3`.
- `x_invisible = truth_tau - selected_visible_tau`.
- Target features are `energy, pt, eta, phi`.
- Invisible `energy` and `pt` use linear `normalize`, not `log_normalize`, so unusual negative-mass debug cases are not blocked by log scaling.
- Truth tau naming is resolved in this order: central `truth_tau_a_p4/truth_tau_b_p4`, raw `GenPart_*` tau branches, then legacy `truth_tau_*` component fields.

The builder also writes source provenance:

- `source_sample_index`
- `source_event_index`
- `source_event_key`

These fields are required because EveNet preprocessing can shuffle and split events. Downstream export must not rely on a local converted-parquet row index.

### Slot Convention

EveNet training slots are not central `lead_a/lead_b` slots. The ML input first identifies tau- and tau+ using tau charge, then canonicalizes the two visible/invisible slots by visible-particle kind:

```text
electron -> muon -> pion -> rho -> other
```

Therefore `x_invisible[:, 0]` and `x_invisible[:, 1]` are particle-kind slots, not fixed tau+ or tau- slots. This gives stable targets for channels such as `e rho`, `mu pi`, and `pi rho`.

When exporting back to central/QI/unfolding, `export_evenet_prediction_to_qi.py` uses `tau_vis_prong_slot*` from the prediction parquet and `lead_a_visible_p4/lead_b_visible_p4` from the central parquet to perform visible-p4 `deltaR` matching. It then restores the central convention:

```text
lead_a = tau+
lead_b = tau-
```

If the central source parquet lacks `lead_a_visible_p4` or `lead_b_visible_p4`, export fails instead of silently using EveNet slot order.

## 2. EveNet Preprocessing

Common split mode:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --file /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/evenet_input.npz \
  --split_ratio 0.4,0.1,0.5 \
  --store_dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_convert
```

Outputs:

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `normalization.pt`
- `shape_metadata.json`

For data-only inference conversion, a practical workaround is:

```bash
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --train /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/data.npz \
  --test /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/data.npz \
  --store_dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_data
```

Use this only as an inference conversion workaround. Do not treat it as a formal train/test split.

## 3. Train Scratch or Pretrain

Scratch:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/scripts/train.py config/train.yaml
```

Pretrain finetuning:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/scripts/train.py config/train_pretrain.yaml
```

Check these paths before launching:

- `platform.data_parquet_dir`
- `options.Dataset.normalization_file`
- `options.Training.model_checkpoint_save_path`
- `options.Training.pretrain_model_load_path`, only for the pretrain config

Prediction uses EMA weights by default when the training config has:

```yaml
EMA:
  enable: true
  replace_model_after_load: true
```

Pass `--disable-ema` during prediction if non-EMA weights are needed.

## 4. Standalone EveNet Prediction

`util/predict_evenet_from_raw_parquet.py` currently consumes converted parquet files such as `test.parquet` or `data.parquet`. It no longer reads raw parquet files or reruns raw-side selection during prediction.

Scratch example:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/predict_evenet_from_raw_parquet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train.yaml \
  --checkpoint /pscratch/sd/t/tihsu/database/ZtautauAnalysis/checkpoint/scratch/last.ckpt \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch \
  --converted-parquet \
    /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_convert/test.parquet \
    /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_data/test.parquet \
  --converted-split-fraction 0.5 \
  --batch-size 2048 \
  --num-gpus 4
```

Pretrain example:

```bash
python3 util/predict_evenet_from_raw_parquet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --checkpoint /pscratch/sd/t/tihsu/database/ZtautauAnalysis/checkpoint/pretrain/last.ckpt \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-pretrain \
  --converted-parquet \
    /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_convert/test.parquet \
    /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_data/test.parquet \
  --converted-split-fraction 0.5 \
  --batch-size 2048 \
  --num-gpus 4
```

`--num-gpus` is batch/event-chunk parallel. A single large parquet can be split across multiple GPUs.

`--converted-split-fraction` only affects `evenet_weight` in the prediction parquet, which is used by standalone data/MC plots. For example, if `test.parquet` represents 50 percent of the MC sample, pass `0.5` so MC `evenet_weight` is scaled by 2. Data is not split-reweighted.

For training/validation alignment checks, use:

```bash
--unweighted-output
```

Do not use this for physics data/MC comparisons unless unit-weight MC is explicitly intended.

Prediction parquet files are self-contained for downstream plotting and export. They include:

- Classification prediction and truth labels
- Predicted invisible slots
- Target invisible truth slots
- Visible tau slots
- `event_weight` and `evenet_weight`
- `source_sample_index`, `source_event_index`, and `source_event_key`

## 5. Standalone EveNet Summary Plots

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/plot_evenet_prediction_summary.py \
  --analysis-config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --mc-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch/test__evenet_pred.parquet \
  --data-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch/data__evenet_pred.parquet \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch/analysis-summary \
  --unblind
```

Typical outputs:

- Weighted classification confusion matrix
- Row-normalized confusion matrix
- Predicted-channel purity stacked yield plot
- Data/MC predicted-channel comparison
- Neutrino truth vs prediction in `E, px, py, pz`
- Neutrino truth vs prediction in `energy, pt, eta, phi`
- Region kinematics plots with reconstructed tau, visible tau, and invisible tau
- `summary_metrics.yaml`

This summary is an EveNet standalone diagnostic. Its regions can be based on EveNet predicted class and do not need to match central cut-based regions.

## 6. Export EveNet Prediction to Central/QI Schema

This step maps EveNet predictions back into the full raw event universe so central/QI/unfolding can consume them directly. Events outside the EveNet input selection, or events without a prediction, are preserved with:

- `flags_valid = false`
- Invalid/default missing p4
- `mmc_likelihood = 0`

Scratch export:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/export_evenet_prediction_to_qi.py \
  --analysis-config config/analysis.yaml \
  --mc-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch/test__evenet_pred.parquet \
  --data-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-scratch/data__evenet_pred.parquet \
  --prediction-split-fraction 0.5 \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export \
  --qi-method-label evenet_scratch
```

Pretrain export:

```bash
python3 util/export_evenet_prediction_to_qi.py \
  --analysis-config config/analysis.yaml \
  --mc-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-pretrain/test__evenet_pred.parquet \
  --data-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/predict-evenet-pretrain/data__evenet_pred.parquet \
  --prediction-split-fraction 0.5 \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export \
  --qi-method-label evenet_pretrain
```

Output structure:

```text
/pscratch/.../qi-evenet-export/
  evenet_scratch/
    data94/filtered___raw.parquet
    Ztautau/filtered___raw.parquet
    Zll/filtered___raw.parquet
    Zqq/filtered___raw.parquet
  evenet_pretrain/
    ...
```

`--prediction-split-fraction` affects the central/QI export `weight` field. For example, if prediction only ran on the test half, pass `0.5`. The adapter scales only MC rows with EveNet predictions by `1 / fraction`. Raw-only rows without EveNet predictions keep their original central weight.

The export preserves central fields such as:

- `lead_a_visible_p4`
- `lead_b_visible_p4`
- Region cut flags
- `event_category`
- `weight`
- MC truth QI fields

The export replaces or adds reconstruction-facing fields:

- `lead_a_missing_p4`
- `lead_b_missing_p4`
- `reco_tau_a_p4`
- `reco_tau_b_p4`
- `flags_valid`
- `mmc_likelihood`
- `theta_cm` and other QI observables
- `evenet_has_prediction`
- `evenet_slot_for_a`
- `evenet_slot_for_b`
- `evenet_leg_match_deltaR_a`
- `evenet_leg_match_deltaR_b`
- `neutrino_method`

EveNet converted slots are not central `lead_a/lead_b`. The adapter uses visible-object matching to map EveNet neutrino slots to central tau legs and avoid a tau+ / tau- semantic swap.

## 7. Central/QI Evaluation

To run the central QI/unfolding framework on EveNet output, point the central config input/output tree at the exported method directory. For example:

```text
default_output_dir: /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_scratch
```

The baseline tree is the normal central output:

```text
/pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline
```

With this layout, the central framework does not need to know whether neutrinos came from MMC, the algebraic solution, EveNet-Pretrain, or EveNet-Scratch. It only sees a consistent parquet schema.

## 8. Baseline vs EveNet-Pretrain vs EveNet-Scratch

`plot_qi_method_comparison.py` supports multiple central-schema method trees:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/plot_qi_method_comparison.py \
  --method Baseline:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline \
  --method EveNet-Pretrain:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_pretrain \
  --method EveNet-Scratch:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_scratch \
  --sample-name Ztautau \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/final-method-comparison
```

Outputs:

- `qi_method_metric_summary.png`
- `neutrino_truth_vs_pred_<region>.png`
- `cut_based_vs_evenet_region_matrix.png`, when the raw parquet contains EveNet predicted class labels
- `qi_method_comparison_metrics.json`

Baseline means the central traditional reconstruction:

- `ee`, `mumu`, and `emu` use MMC.
- Other non-MMC regions use the central algebraic neutrino solution.

EveNet methods come from `export_evenet_prediction_to_qi.py` method trees.

## Weighting Rules

Do not mix the two split-weight corrections:

- `predict_evenet_from_raw_parquet.py --converted-split-fraction`: writes `evenet_weight` in the prediction parquet for standalone EveNet plots.
- `export_evenet_prediction_to_qi.py --prediction-split-fraction`: writes the central/QI export `weight` for central/QI/unfolding.
- Data is never MC split-reweighted.
- If the MC test split is 0.5, use fraction `0.5` to recover full MC yield in data/MC comparisons.
- Raw-only rows without EveNet predictions do not receive the split factor.

## Alignment Checklist

If standalone prediction looks worse than training validation plots, check:

- The prediction command used the intended checkpoint and config.
- EMA usage matches the intended evaluation. Prediction uses EMA by default unless `--disable-ema` is passed.
- Standalone neutrino prediction is conditioned on predicted class. Training validation monitoring may be conditioned on truth class.
- The prediction parquet is regenerated and includes `source_sample_index/source_event_key`.
- `analysis.yaml` `input_files` and `raw_files` come from the same central production.
- `--converted-split-fraction` and `--prediction-split-fraction` are only used in their intended steps.
- The comparison uses the same sample, central schema, and luminosity normalization.

## FAQ

### Why can old prediction parquet files fail export?

They may not contain `source_sample_index` and `source_event_key`. After preprocessing shuffle or train/test splitting, those fields are required to map predictions safely back to raw parquet. Regenerate the EveNet input, preprocess outputs, and prediction parquet.

### Does data without truth fail?

No. Prediction and export allow data without truth fields. Truth metrics are computed only for MC.

### What happens to raw events outside EveNet selection?

The export keeps them with invalid defaults. This is important for unfolding, because truth-region events that fail reconstruction still need to enter the response matrix as missed events.

### Do EveNet regions need to match cut-based regions?

No. Central/QI export `filtered___{region}.parquet` files use central cut flags. Standalone EveNet summaries can use predicted class regions separately.

### Can Pretrain, Scratch, and Baseline be compared together?

Yes. Export each EveNet run with a different `--qi-method-label`, then pass each central-schema tree to `plot_qi_method_comparison.py --method Label:path`.
