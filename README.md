# ML Pipeline

This directory contains the Ztautau EveNet workflow used to go from central
baseline parquet files to EveNet predictions and QI/unfolding-ready inputs.

The README covers only the top-level `ml_pipeline` workflow:

- build EveNet input shards from selected central parquet files
- generate the EveNet event schema from the analysis configuration
- preprocess shards into EveNet train/validation/test parquet files
- train EveNet classification and invisible-particle generation models
- run prediction on converted parquet files
- export predictions to the central QI/unfolding layout
- make the standard validation and summary plots

## Directory Layout

| Path | Purpose |
| --- | --- |
| `config/analysis.yaml` | Sample list, input parquet paths, raw parquet paths, luminosity, class labels, regions, and feature layout. |
| `config/evenet_schema.yaml` | EveNet process/generation schema used to build `generated_event_info.yaml`. |
| `config/preprocess_config.yaml` | EveNet preprocessing config. |
| `config/train_*.yaml` | EveNet training configs for classification and diffusion/generation models. |
| `build_evenet_input_from_parquet.py` | Convert central selected parquets into EveNet input shards. |
| `generate_event_info_yaml.py` | Generate EveNet event-info YAML and a small JSON summary. |
| `preprocess_evenet_parquet.py` | Convert input shards into EveNet train/validation/test files and normalization metadata. |
| `predict_evenet.py` | Run EveNet inference on preprocessed/converted parquet files. |
| `export_evenet_qi_inputs.py` | Export prediction parquets to the central processed-parquet layout for QI/unfolding. |
| `monitor_input.py` | Optional input-level monitoring plots. |
| `monitor_method_comparison.py` | Optional method-comparison monitoring plots. |
| `plot_channel_purity_side_by_side.py` | Optional channel yield, purity, and significance comparison plot. |
| `extract_qi_calibration_magnitude.py` | Optional calibration-shift summary after QI export. |
| `extract_qi_final_measurements.py` | Optional parser and plotter for central `results.txt` files. |
| `extract_response_matrix_summary.py` | Optional response-matrix summary from central ROOT outputs. |

## Environment

Run commands from the repository root unless a command explicitly says
otherwise.

EveNet should be installed from the `tautau` branches below:

- `EveNet-Full`: <https://github.com/tihsu99/EveNet-Full/tree/tautau>
- `Core`: <https://github.com/tihsu99/Core/tree/tautau>

The `Core` repository provides the `evenet` Python package and must live at
`ml_pipeline/EveNet-Full/evenet`.

For a fresh setup, clone both repositories into the expected layout:

```bash
cd /path/to/lep_tree_ana/ml_pipeline

git clone --branch tautau --single-branch \
  https://github.com/tihsu99/EveNet-Full.git \
  ml_pipeline/EveNet-Full

git clone --branch tautau --single-branch \
  https://github.com/tihsu99/Core.git \
  ml_pipeline/EveNet-Full/evenet
```

After installation, expose the local `ml_pipeline` helpers and the vendored
EveNet source tree:

```bash
export PYTHONPATH="$PWD/ml_pipeline:$PWD/ml_pipeline/EveNet-Full:$PYTHONPATH"
```

Verify the install before starting a campaign:

```bash
python3 -c "import evenet, awkward, torch; print(torch.__version__)"
evenet-train --help
```

Training currently requires `WANDB_API_KEY` to be set because the EveNet
training entry point initializes a Weights & Biases logger:

```bash
export WANDB_API_KEY="<your-key>"
```

## Configure a Campaign

Before running a new production, edit `ml_pipeline/config/analysis.yaml`.

Check these fields carefully:

- `Samples.<sample>.input_files`: selected central parquet files, usually
  `filtered___baseline.parquet`.
- `Samples.<sample>.raw_files`: central raw parquet files, usually
  `filtered___raw.parquet`; needed when exporting QI inputs.
- `Samples.<sample>.is_data` and `is_signal`: used for class labels, targets,
  truth fields, and event weights.
- `Samples.<sample>.lumi`: data luminosity in pb.
- `Samples.<sample>.norm_factor`: MC cross section or normalization factor in pb.
- `Inputs.Part` and `Inputs.Global`: the features written into EveNet input
  tensors.
- `Subcategories` and `NeutrinoPrediction`: signal channel labels and channels
  where invisible-particle prediction is trained/evaluated.

For training, also edit the relevant `ml_pipeline/config/train_*.yaml` files:

- `platform.data_parquet_dir`
- `platform.data_parquet_val_dir`
- `options.Training.model_checkpoint_save_path`
- `options.Training.model_checkpoint_load_path`
- `options.Training.pretrain_model_load_path`
- `options.Dataset.normalization_file`
- `logger`

## End-to-End Workflow

The examples below use shell variables to keep paths readable:

```bash
export CAMPAIGN_DIR=/path/to/campaign
export INPUT_DIR="$CAMPAIGN_DIR/evenet-input-shards"
export PREPROCESS_DIR="$CAMPAIGN_DIR/evenet-input"
export PRED_DIR="$CAMPAIGN_DIR/evenet-prediction"
export QI_DIR="$CAMPAIGN_DIR/qi-study"
```

### 1. Generate EveNet Event Info

Generate `generated_event_info.yaml` after changing sample labels, feature
definitions, or the EveNet schema.

```bash
python3 ml_pipeline/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline/config/generated_event_info.yaml
```

Outputs:

- `ml_pipeline/config/generated_event_info.yaml`
- `ml_pipeline/config/generated_event_info.summary.json`

Keep this file aligned with `analysis.yaml`. The class order in this file must
match the class order used when building input shards and training.

### 2. Build EveNet Input Shards

Convert central selected parquet files into EveNet-ready shards.

```bash
python3 ml_pipeline/build_evenet_input_from_parquet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --output-dir "$INPUT_DIR" \
  --batch-size 50000 \
  --rows-per-shard 100000 \
  --num-workers 4
```

Optional sample subset:

```bash
python3 ml_pipeline/build_evenet_input_from_parquet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --output-dir "$INPUT_DIR" \
  --samples Ztautau Zll Zqq \
  --num-workers 4
```

Outputs:

- `$INPUT_DIR/shards/<sample>/*.parquet`
- `$INPUT_DIR/monitoring/<sample>/*.png`
- `$INPUT_DIR/monitoring/comparison/*.png`
- `$INPUT_DIR/manifest.json`

Important behavior:

- Data events receive unit event weight.
- MC event weights use `lumi * norm_factor / sum(initial_total_num_events)`.
- Signal samples get truth invisible targets and truth angular observables.
- Data samples get classification index `-1`.
- `manifest.json` is the input to the preprocessing step.

### 3. Preprocess for EveNet

Preprocess the input shards into train, validation, test, and data parquet
files. The default split is `0.4,0.1,0.5`.

```bash
python3 ml_pipeline/preprocess_evenet_parquet.py \
  --manifest "$INPUT_DIR/manifest.json" \
  --config ml_pipeline/config/preprocess_config.yaml \
  --store-dir "$PREPROCESS_DIR" \
  --split-ratio 0.4,0.1,0.5 \
  --num-workers 4 \
  --verbose
```

Outputs:

- `$PREPROCESS_DIR/train/*.parquet`
- `$PREPROCESS_DIR/val/*.parquet`
- `$PREPROCESS_DIR/test/*.parquet`
- `$PREPROCESS_DIR/train-diffusion/*.parquet`
- `$PREPROCESS_DIR/val-diffusion/*.parquet`
- `$PREPROCESS_DIR/test-diffusion/*.parquet`
- `$PREPROCESS_DIR/data/*.parquet`
- `$PREPROCESS_DIR/shape_metadata.json`
- `$PREPROCESS_DIR/normalization.pt`
- `$PREPROCESS_DIR/preprocess_manifest.json`

Use `train`, `val`, and `test` for classification. Use
`train-diffusion`, `val-diffusion`, and `test-diffusion` for invisible-particle
generation. The diffusion splits keep only events with valid invisible targets.

### 4. Train EveNet Models

Update the chosen train config so that:

- classification configs point to `$PREPROCESS_DIR/train` and
  `$PREPROCESS_DIR/val`
- diffusion configs point to `$PREPROCESS_DIR/train-diffusion` and
  `$PREPROCESS_DIR/val-diffusion`
- `options.Dataset.normalization_file` points to
  `$PREPROCESS_DIR/normalization.pt`
- checkpoint save/load paths point to your campaign directories

Classification examples:

```bash
evenet-train ml_pipeline/config/train_pretrain_cls.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/pretrain-cls"

evenet-train ml_pipeline/config/train_scratch_cls.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/scratch-cls"
```

Diffusion/generation examples:

```bash
evenet-train ml_pipeline/config/train_pretrain.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/pretrain-diffusion"

evenet-train ml_pipeline/config/train_scratch.yaml \
  --ray_dir "$CAMPAIGN_DIR/ray/scratch-diffusion"
```

The best checkpoints are written under
`options.Training.model_checkpoint_save_path` in each train config.

### 5. Run EveNet Prediction

Run prediction on the converted test split. Pass both checkpoints when
classification and diffusion were trained separately.

```bash
python3 ml_pipeline/predict_evenet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --train-config ml_pipeline/config/train_pretrain.yaml \
  --classification-checkpoint /path/to/classification/best.ckpt \
  --diffusion-checkpoint /path/to/diffusion/best.ckpt \
  --converted-parquet "$PREPROCESS_DIR/test" \
  --shape-metadata "$PREPROCESS_DIR/shape_metadata.json" \
  --output-dir "$PRED_DIR/mc" \
  --converted-split-fraction 0.5 \
  --batch-size 8192 \
  --num-gpus 4
```

Run data prediction separately:

```bash
python3 ml_pipeline/predict_evenet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --train-config ml_pipeline/config/train_pretrain.yaml \
  --classification-checkpoint /path/to/classification/best.ckpt \
  --diffusion-checkpoint /path/to/diffusion/best.ckpt \
  --converted-parquet "$PREPROCESS_DIR/data" \
  --shape-metadata "$PREPROCESS_DIR/shape_metadata.json" \
  --output-dir "$PRED_DIR/data" \
  --batch-size 8192 \
  --num-gpus 4
```

Useful options:

- `--task-num-shards` and `--task-shard-index`: split a prediction campaign
  across independent scheduler jobs.
- `--skip-merge`: keep per-chunk `*.part*.parquet` outputs.
- `--merge-only`: merge existing part files without rerunning inference.
- `--delete-merged-parts`: remove part files after a successful merge.
- `--use-truth-classification`: use the truth class stored in the converted
  parquet instead of running the classification checkpoint.

Prediction outputs are written as:

- `<input_stem>__evenet_pred.parquet`
- optional `<input_stem>__evenet_pred.partNNN.parquet` chunk files

### 6. Export Prediction to QI Inputs

Export EveNet prediction parquets to the central processed-parquet layout used
by QIProcessor and ForwardFoldingProcessor.

```bash
python3 ml_pipeline/export_evenet_qi_inputs.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --prediction-parquet "$PRED_DIR/mc/*__evenet_pred.parquet" "$PRED_DIR/data/*__evenet_pred.parquet" \
  --base-dir "$QI_DIR" \
  --methods evenet \
  --num-workers 4 \
  --batch-size 50000
```

Use `--mc-split-fraction` only if the MC prediction parquets were produced from
a split and the prediction step did not already apply `1 / split_fraction`.
If prediction was run with `--skip-merge`, pass the part-file directories or a
part-file glob instead.

Outputs:

- `$QI_DIR/evenet/processed/<sample>/filtered___raw.parquet`
- `$QI_DIR/evenet/processed/<sample>/filtered___<region>.parquet`
- `$QI_DIR/evenet/processed/<sample>/cutflow_<sample>.json`
- `$QI_DIR/evenet/config_evenet.yaml`
- `$QI_DIR/export_summary.json`

The generated `config_evenet.yaml` can be passed directly to the central
`tree_ana` workflow.

### 7. Run Central QI/Unfolding

From the central analysis environment:

```bash
python3 bin/tree_ana \
  -c "$QI_DIR/evenet/config_evenet.yaml"
```

The central run writes QI and response-matrix outputs under the `run` directory
configured by `config_evenet.yaml`.

## Optional Validation and Summaries

### Monitor EveNet Inputs

```bash
python3 ml_pipeline/monitor_input.py \
  --data-dir "$INPUT_DIR/shards/data94" \
  --mc-dir "$INPUT_DIR/shards/Ztautau" "$INPUT_DIR/shards/Zll" "$INPUT_DIR/shards/Zqq" \
  --config ml_pipeline/config/analysis.yaml \
  --output-dir "$CAMPAIGN_DIR/monitor-input" \
  --num-workers 4
```

### Compare Exported Channel Purity

```bash
python3 ml_pipeline/plot_channel_purity_side_by_side.py \
  --method EveNet:"$QI_DIR/evenet" \
  --baseline-xlsx data/baseline_yield.xlsx \
  --output "$CAMPAIGN_DIR/channel_purity_side_by_side.png"
```

Add `--unblind` only when data/MC panels should be shown.

### Summarize Calibration Magnitude

```bash
python3 ml_pipeline/extract_qi_calibration_magnitude.py \
  --method EveNet:"$QI_DIR/evenet" \
  --output-prefix "$CAMPAIGN_DIR/calibration_magnitude"
```

Outputs:

- `<prefix>.json`
- `<prefix>.csv`
- `<prefix>.png`
- `<prefix>.pdf`

### Extract Final QI Measurements

```bash
python3 ml_pipeline/extract_qi_final_measurements.py \
  --method EveNet:"$QI_DIR/evenet/run/QI_analysis/results.txt" \
  --output-prefix "$CAMPAIGN_DIR/qi_results/evenet"
```

For multiple methods:

```bash
python3 ml_pipeline/extract_qi_final_measurements.py \
  --method Baseline:/path/to/baseline/results.txt \
  --method EveNet:"$QI_DIR/evenet/run/QI_analysis/results.txt" \
  --output-prefix "$CAMPAIGN_DIR/qi_results/baseline_vs_evenet"
```

Outputs:

- `<prefix>_per_channel.csv`
- `<prefix>_per_channel.json`
- `<prefix>_combined.csv`
- `<prefix>_combined.json`
- comparison plots unless `--no-plots` is passed

### Summarize Response Matrices

```bash
python3 ml_pipeline/extract_response_matrix_summary.py \
  --method EveNet:"$QI_DIR/evenet/run/ForwardFoldingProcessor/response_matrices" \
  --output-prefix "$CAMPAIGN_DIR/response_matrix/evenet"
```

Outputs include per-matrix metric tables, summary plots, and per-observable
matrix grids.

## Troubleshooting

### `ModuleNotFoundError: evenet`

Set `PYTHONPATH` from the repository root:

```bash
export PYTHONPATH="$PWD/ml_pipeline:$PWD/ml_pipeline/EveNet-Full:$PYTHONPATH"
```

or install the vendored EveNet package:

```bash
python3 -m pip install -e ml_pipeline/EveNet-Full
```

### Class Labels Do Not Match

Regenerate `generated_event_info.yaml` whenever `Samples`, `Subcategories`, or
`NeutrinoPrediction` changes:

```bash
python3 ml_pipeline/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline/config/generated_event_info.yaml
```

Then rebuild input shards and rerun preprocessing.

### Prediction Weights Look Too Small

If prediction used only a split of the MC sample, ensure exactly one of these
steps applies the split correction:

- prediction: `--converted-split-fraction 0.5`
- export: `--mc-split-fraction 0.5`

Do not apply both for the same MC prediction files.

### Export Cannot Find Raw Complement Events

Check that every sample in `analysis.yaml` has valid `raw_files`. The QI export
uses these files to rebuild the raw complement outside the selected baseline
rows.
