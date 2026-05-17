# ML Pipeline

This directory contains the ML pipeline scripts for the Ztautau analysis, including:
- Build EveNet input parquet files from the baseline-selected parquet files
- Generate the `generated_event_info.yaml` schema for EveNet training and prediction
- Preprocess the EveNet input parquet files into train/val/test splits
- Run EveNet prediction on the converted parquet files
- Export the EveNet prediction into the central QI/unfolding parquet layout
- Produce the pre-unfolding validation plots from the exported central-schema parquet trees
- Run the unfolding / `tree_ana` preload workflow from the lite framework path
- Extract the final QI measurements from `results.txt`, write JSON/CSV tables, and draw the per-channel comparison plots
- Draw the prediction summary plots from the prediction parquet outputs



## `build_evenet_input_from_parquet.py`

This rewrite keeps only the fields needed for later EveNet work:

- filtered per-particle `Part_*` inputs needed by EveNet
- `Global` condition features needed by EveNet
- visible tau `a/b` four-vectors
- truth tau `a/b` four-vectors for `Ztautau` only
- target invisible slot `pt / eta / phi` for `Ztautau` only
- classification target index / class name aligned with the generated EveNet class order
- source slot mapping
- simple event metadata and region cuts
- recomputed truth angular observables

It is designed for large parquet inputs:

- batch-wise parquet streaming
- one worker per input parquet
- bounded-memory parquet shard outputs
- worker-side histogram filling with merged monitoring plots at the end
- `input_files` are used first, so the baseline-selected parquet is the default source
- MC normalization uses the sample-level sum of `initial_total_num_events` across all parquet files in the sample

### Example

```bash
python3 build_evenet_input_from_parquet.py   \
  --analysis-config config/analysis.yaml   \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/  \
  --batch-size 50000  \
  --rows-per-shard 100000 \
  --num-workers 4
```

Outputs:

- `shards/<sample>/*.parquet`
- `monitoring/<sample>/*.png`
- `monitoring/comparison/*.png` for data vs stacked MC control plots
- `manifest.json`

The output parquet stores:

- the filtered sequential EveNet inputs as `Part_*` jagged columns
- the global condition inputs as flat scalar columns
- `classification`
- `classification_target_index`
- `classification_target_name`

with the same class ordering used by `generated_event_info.yaml`.

If a required `Global` feature is not stored directly, `missing_p4` is used for
the nominal fallback, including `missing_pz`.


## `generate_event_info_yaml.py`

Generate the `generated_event_info.yaml` schema consumed by:

- `ml_pipeline/config/preprocess_config.yaml`
- `ml_pipeline/config/train.yaml`
- `ml_pipeline/config/train_pretrain.yaml`

Example:

```bash
python3 ml_pipeline_lite/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline_lite/generated_event_info.yaml
```

This writes:

- `generated_event_info.yaml`
- `generated_event_info.summary.json`


## `preprocess_evenet_parquet.py`

Preprocess the lite parquet shards into shuffled train/val/test parquet files
using the standard EveNet preprocessing logic.

This stage is intentionally simple:

- read the lite shard manifest
- use all non-data shards as training input
- optionally preprocess data shards into `store-dir/data`
- event-level split inside each shard
- shuffle rows per written parquet file
- keep `event_weight` unchanged

Example:

```bash
python3 preprocess_evenet_parquet.py  \
  --manifest /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/manifest.json \
  --config config/preprocess_config.yaml  \
  --store-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/evenet_input  \
  --split-ratio 0.4,0.1,0.5  \
  --num-workers 4  \
  --verbose
```

Outputs:

- `train_*.parquet`
- `val/val_*.parquet`
- `test/test_*.parquet`
- `data/data_*.parquet`
- `shape_metadata.json`
- `normalization.pt`
- `preprocess_manifest.json`


## `predict_evenet_from_raw_parquet.py`

Run EveNet prediction on the converted parquet files with the nominal weight rule:

- `evenet_weight = event_weight`
- if a split fraction is provided, scale by `1 / split_fraction`

This lite entry keeps only the converted-parquet workflow and the chunk/shard
controls needed for large-scale production.

Example:

```bash
python3 predict_evenet_from_raw_parquet.py \
  --analysis-config config/analysis.yaml \
  --train-config config/train_pretrain.yaml \
  --classification-checkpoint /path/to/checkpoint-pretrain-cls/best.ckpt \
  --diffusion-checkpoint /path/to/checkpoint-pretrain-diffusion/best.ckpt \
  --converted-parquet /path/to/evenet_input/test \
  --shape-metadata /path/to/evenet_input/shape_metadata.json \
  --output-dir /path/to/prediction-evenet-pretrain \
  --converted-split-fraction 0.5 \
  --batch-size 8192 \
  --num-gpus 4 \
  --task-num-shards 4 \
  --task-shard-index 0
```


## `export_evenet_prediction_to_qi.py`

Export prediction parquet files into the central QI/unfolding parquet layout.

The core behavior is intentionally simple:

- use the prediction-parquet `evenet_weight` directly
- rebuild the predicted baseline-selected rows with calibrated tau and QI observables
- keep the raw complement outside the selected baseline rows and append it with default invalid reco fields
- optionally write a truth-neutrino oracle tree

Example:

```bash
python3 export_evenet_prediction_to_qi.py \
  --analysis-config config/analysis.yaml \
  --mc-pred-parquet /path/to/prediction-evenet-pretrain \
  --data-pred-parquet /path/to/prediction-evenet-pretrain/data-pred \
  --output-dir /path/to/prediction-evenet-pretrain/qi-export \
  --qi-method-label pretrain \
  --write-truth-neutrino-copy \
  --truth-qi-method-label truth \
  --raw-batch-size 50000 \
  --prediction-batch-size 25000 \
  --num-workers 4 \
  --worker-backend thread
```

Outputs:

- `<output-dir>/<qi-method-label>/<sample>/filtered___raw.parquet`
- `<output-dir>/<qi-method-label>/<sample>/filtered___<region>.parquet`
- optionally `<output-dir>/<truth-qi-method-label>/<sample>/...`
- `<output-dir>/<qi-method-label>__qi_export_summary.json`


## `export_evenet_qi_inputs.py`

Export the EveNet prediction parquet files into the central processed-parquet
layout expected by `tree_ana`.

This path is designed to keep the central framework unchanged:

- write `filtered___raw.parquet` and `filtered___<region>.parquet` for each sample
- preserve the fields needed by `QIProcessor`, `ForwardFoldingProcessor`, and `ResponseMatricesManager`
- recompute RAW-complement MC weights from
  `luminosity * norm_factor / sum(initial_total_num_events over raw files)`
- build a central-style config that can run both unfolding and forward folding

`--prediction-parquet` does not identify data / MC from the file name. Each row
must carry `sample_key` or `source_sample_index`, and the final data-vs-MC
decision comes from `Samples.<sample>.is_data` in `ml_pipeline/config/analysis.yaml`.

Example:

```bash
python3 export_evenet_qi_inputs.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --prediction-parquet /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_delta_tau_based/predict-evenet/ \
  --pseudo-data \
  --num-workers 4 \
  --base-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study
```

Outputs:

- `<base-dir>/<method>/processed/<sample>/filtered___raw.parquet`
- `<base-dir>/<method>/processed/<sample>/filtered___<region>.parquet`
- `<base-dir>/<method>/config_<method>.yaml`
- `<base-dir>/export_summary.json`

### Next step: run the central code

The generated `config_<method>.yaml` contains both `QIProcessor` and
`ForwardFoldingProcessor`, so the central `tree_ana` can run directly on the
exported ntuples.

Example for EveNet:

```bash
python3 bin/tree_ana \
  -c /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/evenet/config_evenet.yaml
```

This writes, under:

- `/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/evenet/run/QI_analysis/`
- `/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/evenet/run/ForwardFoldingProcessor/`

If you want to run only one central processor, keep the same generated config
but remove the other block from `Processors` before running:

- keep only `Processors.QIProcessor` to run unfolding only
- keep only `Processors.ForwardFoldingProcessor` to run forward folding only

### Next step: extract QI results

After `tree_ana` finishes, extract the final QI summary from `results.txt`.
Pass the exact `results.txt` path when possible.

```bash
python3 ml_pipeline/util/extract_qi_final_measurements.py \
  --method EveNet:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/evenet/run/QI_analysis/results.txt \
  --output-prefix /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/summary/evenet
```

To compare multiple methods:

```bash
python3 ml_pipeline/util/extract_qi_final_measurements.py \
  --method Baseline:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/baseline/run/QI_analysis/results.txt \
  --method EveNet:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/evenet/run/QI_analysis/results.txt \
  --output-prefix /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-study/summary/baseline_vs_evenet
```

The extractor writes:

- `<prefix>_per_channel.csv`
- `<prefix>_per_channel.json`
- `<prefix>_combined.csv`
- `<prefix>_combined.json`
- comparison plots unless `--no-plots` is given


## `plot_preunfolding_validation.py`

Produce the lite pre-unfolding validation plots from the exported central-schema
parquet trees.

This rewrite keeps only the nominal summary path:

- stored reco observables only
- truth-vs-reco panels
- truth-vs-reco summary plots
- optional data-vs-MC control plots

It intentionally does not run:

- recomputed reco observables
- truth-neutrino upper-limit plots
- missing-neutrino / reco-tau / visible-tau validation branches

Example:

```bash
python3 plot_preunfolding_validation.py \
  --method Baseline:/path/to/baseline/qi-export \
  --method EveNet-Pretrain:/path/to/pretrain/qi-export \
  --signal-sample-name Ztautau \
  --data-sample-name data94 \
  --mc-sample-names Ztautau Zll Zqq \
  --output-dir /path/to/preunfolding-validation \
  --num-workers 8 \
  --load-batch-size 50000
```

Outputs:

- `truth_vs_reco/*.png`
- `truth_vs_reco_summary/*.png`
- `preunfolding_validation_summary.json`
- `preunfolding_validation_report.md`
- optionally the standard data-vs-MC control-plot PNGs


## `run_tree_ana_root_preload.py`

Run the unfolding / `tree_ana` preload workflow from the lite framework path.

This rewrite keeps the preload and region-parallel unfolding control in
`ml_pipeline_lite`, while still launching the shared `tree_ana` executable.

Example:

```bash
python3 run_tree_ana_root_preload.py \
  -c /path/to/config_qi.yaml \
  --num-workers 4 \
  --root-step-size 20000 \
  --raw-batch-size 25000
```


## `extract_qi_final_measurements.py`

Extract the final QI measurements from `results.txt`, write JSON/CSV tables,
and draw the per-channel comparison plots.

This rewrite lives fully in `ml_pipeline_lite` and keeps the
Truth/Reconstruction comparison support.

Example:

```bash
python3 extract_qi_final_measurements.py \
  --method Baseline:/path/to/baseline/results.txt \
  --method EveNet-Pretrain:/path/to/pretrain/results.txt \
  --output-prefix /path/to/qi_compare \
  --keep-truth
```


## `plot_evenet_prediction_summary.py`

Draw the prediction summary plots from the prediction parquet outputs.

This rewrite lives in `ml_pipeline_lite` and defaults the config paths to:

- `ml_pipeline_lite/config/analysis.yaml`
- `ml_pipeline_lite/config/evenet_schema.yaml`

Example:

```bash
python3 plot_evenet_prediction_summary.py \
  --mc-parquet /path/to/prediction-evenet-pretrain \
  --data-parquet /path/to/prediction-evenet-pretrain/data-pred \
  --output-dir /path/to/prediction-evenet-pretrain/summary \
  --weight-source evenet \
  --unblind
```

## Shuffle the processed EveNet input files
This is an optional step to further shuffle the preprocessed EveNet input files. 
This is useful for training stability when the number of input files is small and each file contains a large number of events. 
The shuffling is done by reading the input parquet files in batches and writing out new parquet files with shuffled rows.
```bash
python3 mix_evenet_train_parquets.py \
  --input-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/evenet_input/ \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/evenet_input_shuffled \
  --rows-per-output 100000 \
  --read-batch-size 8192  \
  --seed 42
```
