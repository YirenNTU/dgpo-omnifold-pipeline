# LEP EveNet ml_pipeline

這個目錄是 LEP tau-pair 分析的 ML 側 pipeline。原則是：

- 不修改 central analysis / unfolding framework。
- EveNet 只在 `ml_pipeline` 內完成 input building、training、prediction、plotting。
- 要接回 central/QI/unfolding 時，用 adapter 把 EveNet prediction 寫成 central parquet schema。
- 預設執行位置是這個目錄：`cd ml_pipeline`。

## 目錄角色

- `config/analysis.yaml`: ML pipeline 的主控設定。包含 sample input/raw parquet、feature list、normalisation、prediction/export 預設路徑。
- `config/evenet_schema.yaml`: EveNet process topology、classification class、truth-generation schema。
- `config/preprocess_config.yaml`: EveNet preprocessing wrapper，指向 generated event info。
- `config/train.yaml`: EveNet scratch training config。
- `config/train_pretrain.yaml`: EveNet pretrain finetuning config。
- `util/build_evenet_input_from_parquet.py`: central/DataLoader parquet -> EveNet `.npz`，並產生 monitoring plots。
- `util/predict_evenet_from_raw_parquet.py`: standalone EveNet inference。檔名是 legacy，現在只吃 EveNet converted parquet。
- `util/plot_evenet_prediction_summary.py`: standalone EveNet prediction summary plots。
- `util/export_evenet_prediction_to_qi.py`: EveNet prediction -> central/QI parquet schema。
- `util/plot_qi_method_comparison.py`: Baseline、EveNet-Pretrain、EveNet-Scratch 等多方法比較。
- `util/evenet_parquet_common.py`: visible tau、truth invisible target、source-event matching 共用邏輯。

## 核心資料流

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

## 0. 先產生 central baseline parquet

這一步在 repo root 跑 central framework，不在 `ml_pipeline` 裡改 central code。

```bash
cd /path/to/lep_tree_ana
python3 bin/tree_ana \
  --config-yaml config/config.yaml \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline
```

之後 `ml_pipeline/config/analysis.yaml` 要指到這些輸出：

```yaml
Samples:
  Ztautau:
    input_files:
      - /pscratch/.../baseline/Ztautau/filtered___baseline.parquet
    raw_files:
      - /pscratch/.../baseline/Ztautau/filtered___raw.parquet
```

`input_files` 是 EveNet input universe，通常是已通過 central baseline/preselection 的 events。`raw_files` 是 full raw universe，用於最後接回 central/QI/unfolding，沒有 EveNet prediction 的 event 也會保留下來並填 default invalid value。

## 1. 建 EveNet input

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/build_evenet_input_from_parquet.py \
  --config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset
```

主要輸出：

- `/pscratch/.../dataset/evenet_input.npz`
- `/pscratch/.../dataset/evenet_input_metadata.json`
- `config/generated_event_info.yaml`
- monitoring plots under the output directory

目前 EveNet target invisible 的定義是：

- selected visible tau = prong + `dR < 0.3` photon。
- `x_invisible = truth_tau - selected_visible_tau`。
- target features 是 `energy, pt, eta, phi`。
- `energy` 和 `pt` 在 invisible normalisation 使用 linear `normalize`，不是 `log_normalize`，所以可處理 negative mass / unusual kinematics debug case。
- truth tau naming 會依序讀新的 central `truth_tau_a_p4/truth_tau_b_p4`、raw `GenPart_*` tau、最後才 fallback 到舊的 `truth_tau_*` component 欄位。

builder 也會寫 source provenance：

- `source_sample_index`
- `source_event_index`
- `source_event_key`

這些欄位非常重要，因為 preprocessing 可能 shuffle/split；後面 export 回 raw event universe 時不能靠 local row index。

### Slot convention

EveNet training 的兩個 visible/invisible slots 不是 central `lead_a/lead_b`。ML 端會先用 tau charge 找到 tau- / tau+，再依 visible-particle kind canonicalize slot order：

```text
electron -> muon -> pion -> rho -> other
```

所以 `x_invisible[:, 0]` / `x_invisible[:, 1]` 的語意是「particle-kind slot」，不是固定 tau+ / tau-。這樣 `e rho`、`mu pi`、`pi rho` 這類 channel 的 target 才是穩定的。

回到 central/QI/unfolding 時，`export_evenet_prediction_to_qi.py` 會用 prediction parquet 裡的 `tau_vis_prong_slot*` 和 central parquet 裡的 `lead_a_visible_p4/lead_b_visible_p4` 做 visible-p4 `deltaR` matching，重新寫回 central convention：

```text
lead_a = tau+
lead_b = tau-
```

如果 central source parquet 缺少 `lead_a_visible_p4` 或 `lead_b_visible_p4`，export 會直接停止，而不是 fallback 到 EveNet slot order。

## 2. EveNet preprocessing

常用 split 模式：

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --file /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/evenet_input.npz \
  --split_ratio 0.4,0.1,0.5 \
  --store_dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_convert
```

輸出：

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `normalization.pt`
- `shape_metadata.json`

如果只是在 data inference 需要把 `data.npz` 轉成 parquet，可以用 train/test workaround：

```bash
python3 EveNet-Full/preprocessing/preprocess.py \
  --config config/preprocess_config.yaml \
  --train /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/data.npz \
  --test /pscratch/sd/t/tihsu/database/ZtautauAnalysis/dataset/data.npz \
  --store_dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/evenet_data
```

這只是 inference conversion workaround；不要把它當正式 train/test split。

## 3. Train Scratch 或 Pretrain

Scratch:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/scripts/train.py config/train.yaml
```

Pretrain finetune:

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 EveNet-Full/scripts/train.py config/train_pretrain.yaml
```

請確認 config 裡的路徑：

- `platform.data_parquet_dir`
- `options.Dataset.normalization_file`
- `options.Training.model_checkpoint_save_path`
- `options.Training.pretrain_model_load_path`，只對 pretrain config 需要。

目前 prediction 預設會使用 EMA 權重，只要 training config 中：

```yaml
EMA:
  enable: true
  replace_model_after_load: true
```

如果要強制不用 EMA，prediction 時加 `--disable-ema`。

## 4. EveNet standalone prediction

`util/predict_evenet_from_raw_parquet.py` 現在只吃 converted parquet，例如 `test.parquet` 或 `data.parquet`。它不再讀 raw parquet，也不再在 prediction 階段重做 selection。

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

`--num-gpus` 是 batch/event-chunk parallel。單一大 parquet 會被切 chunk 分到多張 GPU。

`--converted-split-fraction` 只影響 prediction parquet 內的 `evenet_weight`，用於 standalone data/MC plots。例子：如果 `test.parquet` 是 MC 的 50%，傳 `0.5` 會讓 MC `evenet_weight` 乘以 2。Data 不做這個 MC split reweighting。

如果你要比較 training/validation alignment，可以加：

```bash
--unweighted-output
```

但做 physics data/MC comparison 時通常不要加，因為那會把 MC event weight 全部設成 1。

prediction parquet 會自帶後續 plot/export 所需資訊：

- classification prediction/truth
- predicted neutrino slots
- target invisible truth slots
- visible tau slots
- `event_weight` / `evenet_weight`
- `source_sample_index` / `source_event_index` / `source_event_key`

所以 summary plot 不需要再讀 converted parquet。

## 5. Standalone EveNet summary plot

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

常見輸出：

- weighted classification confusion matrix
- row-normalized confusion matrix
- predicted-channel purity stacked yield
- data/MC predicted-channel comparison
- neutrino truth vs prediction in `E, px, py, pz`
- neutrino truth vs prediction in `energy, pt, eta, phi`
- region kinematics plots with reconstructed tau, visible tau, invisible tau
- `summary_metrics.yaml`

這個 summary 是 EveNet standalone diagnostic。它的 region 可以用 EveNet predicted class 來看，不要求跟 central cut-based regions 完全一致。

## 6. Export EveNet prediction 到 central/QI schema

這一步是把 EveNet prediction 放回 full raw event universe，讓 central/QI/unfolding 可以直接吃。沒有通過 EveNet input selection 或沒有 prediction 的 event 會被保留，填：

- `flags_valid = false`
- invalid/default missing p4
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

輸出結構：

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

`--prediction-split-fraction` 是給 central/QI parquet 的 `weight` 用。例子：prediction 只跑 test half，就傳 `0.5`，adapter 只會把有 EveNet prediction 的 MC rows 權重乘以 2。沒有 prediction 的 raw-only rows 保持原本 central weight。

export 會保留 central 欄位，例如：

- `lead_a_visible_p4`
- `lead_b_visible_p4`
- region cut flags
- `event_category`
- `weight`
- MC truth QI fields

export 會替換或新增 reconstruction-facing 欄位：

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

EveNet converted slots 不等於 central `lead_a/lead_b`。adapter 會用 visible-object matching 把 EveNet predicted neutrino slots 對到 central tau legs，避免 tau+ / tau- 語意錯位。

## 7. 用 central/QI framework evaluation

如果要讓 central QI/unfolding framework 跑 EveNet output，讓 central config 的 output/input 指到 export method tree。例如：

```text
default_output_dir: /pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_scratch
```

Baseline 則直接用 central 原本輸出的 baseline tree：

```text
/pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline
```

這樣 central framework 不需要知道 neutrino 是 MMC、algebraic、EveNet-Pretrain、還是 EveNet-Scratch；它只看到一致的 parquet schema。

## 8. Baseline vs EveNet-Pretrain vs EveNet-Scratch 比較

`plot_qi_method_comparison.py` 支援多個 method，只要每個 method 都是 central-schema tree：

```bash
cd /path/to/lep_tree_ana/ml_pipeline
python3 util/plot_qi_method_comparison.py \
  --method Baseline:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/baseline \
  --method EveNet-Pretrain:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_pretrain \
  --method EveNet-Scratch:/pscratch/sd/t/tihsu/database/ZtautauAnalysis/qi-evenet-export/evenet_scratch \
  --sample-name Ztautau \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/final-method-comparison
```

輸出：

- `qi_method_metric_summary.png`
- `neutrino_truth_vs_pred_<region>.png`
- `cut_based_vs_evenet_region_matrix.png`，如果 raw parquet 內有 EveNet predicted class。
- `qi_method_comparison_metrics.json`

這裡的 Baseline 是 central 已有的 traditional reconstruction：

- `ee`, `mumu`, `emu` 使用 MMC。
- 其他 non-MMC regions 使用 central algebraic neutrino solution。

EveNet methods 則來自 `export_evenet_prediction_to_qi.py` 的 method tree。

## 權重規則

不要把兩種權重修正混在一起：

- `predict_evenet_from_raw_parquet.py --converted-split-fraction`: 只寫進 prediction parquet 的 `evenet_weight`，給 standalone EveNet plots 用。
- `export_evenet_prediction_to_qi.py --prediction-split-fraction`: 只改 central/QI export 裡的 `weight`，給 central/QI/unfolding 用。
- Data 不做 MC split reweighting。
- 如果 MC test split 是 0.5，data/MC comparison 需要乘回 full MC yield，所以 fraction 設 `0.5`。
- 沒有 EveNet prediction 的 raw-only rows 不乘 split factor，保留原本 central weight。

## Alignment checklist

如果發現 validation plots 和 standalone prediction 差很多，先檢查：

- prediction 是否用了正確的 checkpoint 和 config。
- EMA 是否符合預期；預設會用 EMA，除非加 `--disable-ema`。
- standalone neutrino prediction 是 conditional on predicted class；training validation monitoring 可能是 conditional on truth class。
- prediction parquet 是否是新的，且包含 `source_sample_index/source_event_key`。
- `analysis.yaml` 的 `input_files` 和 `raw_files` 是否來自同一批 central production。
- `--converted-split-fraction` 和 `--prediction-split-fraction` 是否只用在各自該用的地方。
- comparison 是否在同一個 sample、同一個 central schema、同一個 luminosity normalization 下做。

## 常見問題

### 為什麼 old prediction parquet 不能直接 export？

如果它沒有 `source_sample_index` 和 `source_event_key`，train/test split 或 preprocessing shuffle 後無法安全 map 回 raw parquet。請重跑 input builder、preprocess、prediction。

### data 沒有 truth 會不會壞？

不會。prediction parquet 和 export 都允許 data 沒有 truth。truth metrics 只會在 MC 上算。

### raw event 沒有通過 EveNet selection 怎麼辦？

export 會保留它，填 invalid default。這對 unfolding 很重要，因為 truth-region event failed reconstruction 仍要能作為 missed event 進 response matrix。

### region 要跟 cut-based 完全一致嗎？

central/QI export 的 `filtered___{region}.parquet` 會使用 central cut flags。EveNet standalone summary 可以另外用 predicted class 看 region，不需要硬對齊 cut-based region。

### 可以同時比較 Pretrain、Scratch、Baseline 嗎？

可以。分別 export 成不同 `--qi-method-label`，再用 `plot_qi_method_comparison.py --method Label:path` 一次丟進去。
