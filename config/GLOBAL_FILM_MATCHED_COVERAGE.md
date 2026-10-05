# Global FiLM 對照舊 finetuning 的 joint coverage

只回答一個問題：目前 global-FiLM supervised checkpoint，在**完全沿用舊 coverage panel 與 noise 規則**時，是否改善舊架構 finetuning 未能改善的 joint coverage？這不是 FiLM 與 depth 的單因素實驗，也不宣稱 H4 closure。

固定使用 `8mbugqpq` completed epoch100 保存的 reference：

- `/pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_lr_10pct_seed42/checkpoints/joint_coverage/epoch-0100.json` 與同名 `.pt`。
- 原有 `h4_spike_coverage_1110/panel.pt`，1024 events，不重抽 panel。
- K32、legacy DDIM20、每GPU batch16、16 workers、parallel_chains1。
- 每個 panel position 的 CPU noise stream 為 seed `42017 + position`，與舊 callback 完全相同。**不是較早 K128 replay 的 rank-dependent noise。**
- 500次 whole-event bootstrap；保留全部32 draws，不做best-of-K、reweighting或physics projection。
- Primary：joint radius <1e-4 rad 的 absolute truth-gap change及95% CI。Joint radius定義為max(acoplanarity, acollinearity)。同時檢查 joint-radius W1、四個target marginal W1與invalid-direction counts。

預設新模型：`diffusion_global_film_long_resume/checkpoints/last.ckpt`。程式先讀出真實儲存 epoch/step，拒絕後續 DGPO checkpoint、缺少Fourier／FiLM、額外token-readout和非FP32權重。只保留raw `state_dict`至本次輸出的`raw_policy.pt`，各worker從這份固定快照載入；不安裝EMA、optimizer或classifier。完整keys/shapes/dtypes在模型建構後嚴格核對。

舊 `.pt` 樣本會以原panel重新計算角度和metrics，必須重現舊JSON報告。若舊檔缺失或設定不一致則停止，不會偷偷重抽events或換baseline。

先更新既有遠端 `ml_pipeline`，由使用者在**已有的16-GPU Ray allocation**執行一次：

```bash
cd /global/homes/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/evaluate_global_film_coverage.py
```

這個程式不會提交Slurm或建立allocation。Ray的`TorchTrainer`只作分散式執行，worker沒有training loop／optimizer，僅呼叫既有`JointCoverageValidation.evaluate`。Policy updates與classifier fits均為0。連不上既有cluster或GPU不足時不作單機fallback。

可先在遠端檢查保存檔，不連Ray、W&B或啟動GPU sampling：

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/evaluate_global_film_coverage.py --check-only
```

`--dry-run`只印出設定，不存取remote checkpoint或panel。`--check-only`會檢查artifact/protocol與checkpoint metadata，完整模型key匹配仍在worker建構模型後執行。

預設輸出：`/pscratch/sd/y/yiren/Ztautau/global_film_matched_coverage`。若目錄已存在則拒絕覆寫；重跑時提供新的`--output`與`--run-id`。如要指定已知的supervised checkpoint，使用`--checkpoint /absolute/path/to/epoch=....ckpt`，不會自動挑best／latest。

W&B project：`nu2flow-RL`；ID：`globcov1`；group：`Supervised conditioning coverage`。
Display name：`Does conditioning recover joint coverage? | global FiLM | matched old finetuning panel`。

結果包括 `report.json`、`manifest.json`、raw snapshot、原panel與baseline樣本副本、新generated samples、paired CI，以及CDF／2D角度圖。新模型儲存epoch與舊baseline completed-epoch100分開記錄；這不是equal-training-budget比較。

判讀：primary truth-gap CI整段小於0且joint-radius W1下降，支持此checkpoint在這個panel的coverage改善；仍須呈現marginal／invalid-direction取捨。其餘結果保留為mixed/unresolved，不能以少量hit或零empirical CI宣稱support已解決／不存在。Panel曾被探索使用，並非全新confirmation test。

狀態：prepared；未在NERSC執行、未提交工作，尚無新physics結果。

本機驗證：新evaluator測試19項、既有callback測試3項、DDIM／noise replay測試39項，分別執行，共61項通過。另完成dry-run、Python語法與W&B display-name檢查。尚未讀取NERSC上的真實checkpoint／baseline檔案，也未驗證16-GPU執行；可先用上述`--check-only`驗證保存檔案。
