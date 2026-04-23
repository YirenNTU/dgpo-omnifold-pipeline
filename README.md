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

For this analysis pipeline, the documented prediction commands use non-EMA weights by default via `--disable-ema`. This keeps standalone prediction closer to direct checkpoint validation unless an EMA comparison is explicitly intended.

The predictor can still use EMA weights when the training config has:

```yaml
EMA:
  enable: true
  replace_model_after_load: true
```

To run an EMA prediction intentionally, omit `--disable-ema`.

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
  --disable-ema \
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
  --disable-ema \
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
  --mc-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-scratch/test__evenet_pred.parquet \
  --data-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-scratch/data__evenet_pred.parquet \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-scratch \
  --qi-method-label qi-export \
  --num-workers 4
```

Pretrain export:

```bash
python3 util/export_evenet_prediction_to_qi.py \
  --analysis-config config/analysis.yaml \
  --mc-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/test__evenet_pred.parquet \
  --data-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/data__evenet_pred.parquet \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain \
  --qi-method-label qi-export \
  --num-workers 4
```

Output structure:

```text
/pscratch/.../ml_based/predict-scratch/qi-export/
    data94/filtered___raw.parquet
    data94/filtered___Ztautau_pipi.parquet
    Ztautau/filtered___raw.parquet
    Ztautau/filtered___Ztautau_pipi.parquet
    Zll/filtered___raw.parquet
    Zqq/filtered___raw.parquet
```

For current prediction parquets, the central/QI export reads `evenet_weight` from the prediction parquet and writes it into the central `weight` field for rows with EveNet predictions. This means the MC split correction should already be applied by `predict_evenet_from_raw_parquet.py --converted-split-fraction`. Raw-only rows without EveNet predictions keep their original central weight.

`--num-workers` parallelizes the config-driven export over parent samples and prints progress bars such as `mc export [####----] 1/3 Ztautau`. The default backend is thread-based so large awkward arrays are shared instead of copied into subprocesses. Keep it at `1` if memory pressure is high. Use `--worker-backend process` only when enough memory is available.

For the ML-based QIProcessor configs in the repository, write the export directly to the method-specific root:

```bash
python3 util/export_evenet_prediction_to_qi.py \
  --analysis-config config/analysis.yaml \
  --mc-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/test__evenet_pred.parquet \
  --data-pred-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/data__evenet_pred.parquet \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain \
  --qi-method-label qi-export \
  --num-workers 4
```

This produces files under `/pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/qi-export/{sample}/`. The scratch method uses the same pattern with `predict-scratch`.

The export writes both central cut-based files such as `filtered___hadhad.parquet` and ML dedicated files such as `filtered___Ztautau_pirho.parquet`. The ML dedicated files are selected by `evenet_pred_class_name`, while central files are selected by their original central cut flags.

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

This step runs the central `QIProcessor` on the EveNet-exported parquet trees. It is intentionally separate from the standalone ML summary plots.

The important convention is:

- Central cut-based regions are still available: `baseline`, `hadhad`, `ee`, `mumu`, `emu`.
- ML dedicated QI regions are EveNet predicted fine channels: `Ztautau_pipi`, `Ztautau_pirho`, `Ztautau_pie`, `Ztautau_pimu`, `Ztautau_rhoe`, `Ztautau_rhomu`, `Ztautau_rhorho`, `Ztautau_ee`, `Ztautau_mumu`, `Ztautau_emu`.
- The QIProcessor configs in `config/config_qi_evenet_pretrain.yaml` and `config/config_qi_evenet_scratch.yaml` use the ML dedicated regions, not the central broad regions.

First run the Step 6 export for each method. For the QIProcessor configs below, the exported roots must be:

- Pretrain: `/pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/qi-export`
- Scratch: `/pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-scratch/qi-export`

Each exported root should contain:

```text
/pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/qi-export/
  data94/filtered___raw.parquet
  data94/filtered___Ztautau_pipi.parquet
  ...
  Ztautau/filtered___raw.parquet
  Ztautau/filtered___Ztautau_pipi.parquet
  ...
  Zll/...
  Zqq/...
```

The raw parquet also contains matching cut flags such as `Ztautau_pipi_cut`, which QIProcessor needs when building response matrices.

Then run QIProcessor from the repository root, not from `ml_pipeline`:

```bash
cd /path/to/lep_tree_ana
PYTHONPATH=$PWD python3 bin/tree_ana -c config/config_qi_evenet_pretrain.yaml
PYTHONPATH=$PWD python3 bin/tree_ana -c config/config_qi_evenet_scratch.yaml
```

If ROOT/cppyy crashes during import in the active environment, use the ml_pipeline wrapper
instead of editing `bin/tree_ana`:

```bash
cd /path/to/lep_tree_ana
python3 ml_pipeline/util/run_tree_ana_root_preload.py -c config/config_qi_evenet_pretrain.yaml
python3 ml_pipeline/util/run_tree_ana_root_preload.py -c config/config_qi_evenet_scratch.yaml
```

Expected QIProcessor outputs:

```text
/pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_based/predict-pretrain/qi-export/QI_analysis/
  results.txt
  response_Ztautau_pipi.root
  response_Ztautau_pirho.root
  ...
  Ztautau_pipi/
    plots/
    unfolding/
  Ztautau_pirho/
    plots/
    unfolding/
```

If you want the central broad-region QIProcessor instead, use the normal central config style and regions such as `hadhad`, `ee`, `mumu`, and `emu`. Do not mix that with the ML dedicated configs unless the comparison explicitly calls for it.

For older central-style exports, point the central config input/output tree at the exported method directory. For example:

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
  --data-sample-name data94 \
  --mc-sample-names Ztautau Zll Zqq \
  --metric-grouping evenet-channel \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/final-method-comparison
```

Use `--metric-grouping evenet-channel` to make the final metric summary y-axis follow the actual EveNet predicted classes such as `Ztautau_pipi`, `Ztautau_pirho`, `Ztautau_pie`, and `Ztautau_others`. Use `--metric-grouping region` if you intentionally want central cut-based files such as `hadhad`, `ee`, `mumu`, and `emu`.

Outputs:

- `qi_metric_<observable>.png`, one channel-vs-method plot per metric or QI observable
- `physics_data_mc_<method>/<region>_<observable>.png`, data-vs-stacked-MC physics distributions
- `physics_data_mc_<method>/<region>_<observable>_log.png`, log-y companion plots
- `neutrino_truth_vs_pred_<region>.png`
- `cut_based_vs_evenet_region_matrix.png`, comparing central cut-based regions against EveNet fine predicted channels when the raw parquet contains EveNet predicted class labels
- `qi_method_comparison_metrics.json`
- `qi_method_comparison_audit.json`
- `qi_method_comparison_report.md`

The audit JSON and Markdown report are meant for process validation. They list method input roots, per-channel parquet coverage, event counts, weighted yields, valid fractions, QI acceptance, available metric keys, generated diagnostic plots, and physics data-vs-MC plot coverage.

By default, the data-vs-MC plots use `data94` as data and `Ztautau Zll Zqq` as MC. Override those with `--data-sample-name` and `--mc-sample-names`. Use `--physics-observables` to restrict the plotted observable list.

Baseline means the central traditional reconstruction:

- `ee`, `mumu`, and `emu` use MMC.
- Other non-MMC regions use the central algebraic neutrino solution.

EveNet methods come from `export_evenet_prediction_to_qi.py` method trees.

## Weighting Rules

Do not apply the split-weight correction twice:

- `predict_evenet_from_raw_parquet.py --converted-split-fraction`: writes `evenet_weight` in the prediction parquet for standalone EveNet plots.
- `export_evenet_prediction_to_qi.py`: uses prediction-parquet `evenet_weight` for predicted rows when that field is available.
- `export_evenet_prediction_to_qi.py --prediction-split-fraction`: legacy fallback only for old prediction parquets that do not contain `evenet_weight`.
- Data is never MC split-reweighted.
- If the MC test split is 0.5, pass `--converted-split-fraction 0.5` during prediction to recover full MC yield in data/MC comparisons.
- Raw-only rows without EveNet predictions do not receive the split factor.

## Alignment Checklist

If standalone prediction looks worse than training validation plots, check:

- The prediction command used the intended checkpoint and config.
- EMA usage matches the intended evaluation. The documented workflow uses `--disable-ema`; omit it only for an explicit EMA prediction.
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
