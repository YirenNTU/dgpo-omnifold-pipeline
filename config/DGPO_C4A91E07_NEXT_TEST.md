**c4a91e07 後續測試建議 — 2026-09-13**

**2026-09-14 已實作的兩階段實驗**

第一階段固定 c4a91e07 step1110 policy，直接重用 v5 的四個 H4 reward members、independent H4 judge、共同 event panel 與 candidate logits。它不重訓 classifier，也不執行 optimizer step；只沿 calibrated-LOO gradient 以共同 rollout noise 重播 `1e-7, 3e-7, 1e-6, 3e-6, 1e-5` 五個 RMS 半徑的正負方向。嚴格 joint gate 仍記錄 independent-judge AUC gap、response mean absolute bin offset、平均 physics JSD 與 paired response bootstrap，用來顯示局部 trade-off。Pilot 的步長另由 training-purpose rule 選擇：judge gap 和平均 JSD 必須改善、gradient cosine 必須至少 0.99，再取最大半徑。這避免用單一局部 response 變化否決可能震盪或在 fresh reward rounds 後反轉的完整 trajectory。

```bash
shifter python3 scripts/diagnose_reward_epsilon_sweep.py \
  config/dgpo_10pct_c4a91e07_h4_epsilon_sweep.yaml
```

輸出寫到 `/pscratch/sd/y/yiren/Ztautau/c4a91e07_h4_epsilon_sweep_v1/report.json`，並即時記錄在 W&B run `c4a91e07_h4_calibrated_loo_epsilon_sweep_v1`。

第二階段由同一個 c4a91e07 `last.ckpt` 做 weights-only 啟動。它讀取第一階段選出的 training-purpose epsilon，把 policy LR 設為該值、PET/body LR 設為 0.1 倍，並以 v5 temperature 的倒數 temper reward。固定跑五輪，每輪十個 DGPO steps；每輪 reward 都是全新初始化的 H4 `2 repeats × 2 folds` ensemble，classifier optimizer 不繼承，policy optimizer 也在 reward install 後清除。它保留 coefficient=1 的 soft velocity-MSE reference loss，關閉 hard trust、projection、rollback、metric-based stopping 與 checkpoint selection。第五輪結束在預先指定的 step50，不會額外訓練第六組 classifier。Classifier AUC 是 trajectory diagnostic；主要判斷是 config 中預先列出的 topology JSD 與四個 target residual，在 step0、10、20、30、40、50 的整體走勢和固定 step50 endpoint。

```bash
shifter python3 scripts/train_dgpo_h4_fresh_ensemble_pilot.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_5round.yaml
```

Pilot 使用固定 W&B id `h4fr5r01` 與 `resume: allow`，所以 NERSC preemption 後會接回同一個 run。Policy、validation、response、raw monitor 與每個 classifier member 的 loss/BA/AUC 都會即時上傳；若 W&B runtime 不可用、scheduled reward fit 未通過 signal/saturation gate、epsilon report 不合約，或 source/final checkpoint step 不符，launcher 會 fail closed。

建議測試 nonlinear phi-H4 Fourier classifier 的 ensemble 與每輪重新初始化，但先用固定 policy 驗證 density-ratio 品質，再做短 DGPO pilot。Ensemble 是降低估計變異的假說；fresh start 是排除跨輪權重繼承的假說。兩者都不能直接保證 response matrix 改善。

使用者的觀察是：舊方法改善 marginal distributions **和 correlations**，但 response matrix 效果有限。這份調查保留這個區別。此次讀取 W&B 設定、histories 和本地程式；沒有取得 response ROOT files，沒有重新計算 response matrix，也沒有啟動 GPU 訓練或修改現有訓練設定。

**實際實驗證據**

| 實驗 | 讀到的結果 | 對下一步的含義 |
| --- | --- | --- |
| [c4a91e07](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/c4a91e07) | 每輪 reward classifier 已經 cold start：`warm_start_iterations: []`、`warm_start_from_iteration_one: false`；raw monitor 也不繼承。Raw AUC 在 policy step 50 為 0.59993，step 1100 為 0.50524。最後一次 raw fit 只有 50 updates。 | 已做過 fresh start；低 AUC 不能單独排除容量或訓練不足，也不能证明 physics closure。 |
| 同一 run 的 policy validation | 從 epoch -1 到 109，tau-a delta-theta JSD 0.03459→0.01320，tau-b 0.04011→0.01346；cos-opening 0.00894→0.01220，delta-phi-to-pi 0.04058→0.04421。 | 可確認不同 observable 的改善不一致。這些是 distribution metrics，不能冒充 response-matrix 結果，也沒有量化使用者看到的 correlation 改善。 |
| [b8e2f04h](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/b8e2f04h) | 從 c4a91e07 的 step-1110 policy 完整 resume。第一個 raw check 在 step 1110、尚無新 policy update 時 AUC=0.89279；step 1150 為 0.89762。兩次 `omnifold/accepted` 都是 0。 | H4 能看見同一 source policy 的殘餘差異。這兩次整套新 reward 沒有被接受；不要把這段 physics 幾乎不變當成已成功安裝 Fourier reward 後的效果。Residual member 的 `accepted=1` 與整套 reward 接受不同。 |
| 同一 H4 run 的訓練曲線 | 第一段 raw fit 中，首筆 AUC>0.55 的記錄在 fit step 520；兩次完整 raw fit 分別跑了 2460、2010 updates。 | Cold fit 若在 50–300 updates 因 near-chance plateau 停下，容易錯過較晚出現的學習。最小訓練量與 saturation 必須一起看。 |
| [30d9dc0f](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/30d9dc0f) | 舊 classifier 的 step260 cold-restart pilot。Raw AUC 起點 0.50673、step70 為 0.51198；delta-phi-to-pi JSD 起點 0.02623、epoch5 為 0.02678。 | 沒有看到 restart 自動帶來該 topology 指標的改善；不是多 seed 的因果比較。 |
| [4267caeb](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/4267caeb) / [1fe91ab2](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/1fe91ab2) | 已接受新 reward 的 Fourier+rest-frame 試驗，delta-phi-to-pi JSD 分別 0.02492→0.07017、0.02346→0.09722；後者 raw AUC 約 0.91。 | 確有「classifier 很強，但 policy physics 變差」的紀錄。增加容量、繼承、延長 residual fitting 都不是充分解法。兩者起點為 f6b4ec46 step320，不能與 c4a91e07 作 matched-start 排名。 |
| [122b9d84](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/122b9d84) | Iteration-1-only 試驗的 delta-phi-to-pi JSD 0.02389→0.08463；最後記錄的 OOF ESS/N 約 0.02491。 | 只保留第一個 residual，也沒有自動解決不穩定。 |
| [c3b347fe4eea](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/c3b347fe4eea) | 固定 step320 policy diagnostic 的 seed20260906、iteration3：fold sign disagreement=48.53%；OOF cumulative ESS/N 在該 proposed increment 前為 0.006948，若套用則為 0.000545（333272 events 只剩 ESS≈182）；top-1% weight mass=88.58%。Outer validation ESS/N 為 0.06542→0.02477。 | 有很強的 fold/weight-path 不一致與尾部集中訊號，值得測 repeated crossfit。但這是未完成 diagnostic 的局部紀錄，且是不同 checkpoint/架構；不是證明 c4a91e07 或 H4 一定有相同根因。 |

這些 run 的 W&B `crashed` 狀態不能直接解讀成數值發散；可能包含中斷或 infrastructure 問題。表中只依據具體 metrics 作判斷。查閱最近 100 個 runs 未找到已完成的 5-fold ensemble 成效報告，不能把 repo 中的 ablation 設計當成實驗成功結果。

可重查的設定及 scalar/history 摘錄在 [experiment_evidence.json](../artifacts/c4a91e07_review/experiment_evidence.json)。Histories 以每個 metric 最多 1500 筆讀取；列出的少量 physics/raw/final acceptance points 未達此上限。W&B summary 各 key 的最後值不一定來自同一時間，因此有 history 時使用 co-logged epoch/global_step。

**使用指定 last.ckpt**

W&B run 的輸出設定對應：

```text
/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/last.ckpt
```

W&B 最後記錄為 policy step1110；後續 H4 fork 的實際 source 是同目錄 `dgpo-epoch=110-next_ep=111-step=1110.ckpt`。這是 step1110 來源的交叉佐證，但此次沒有讀取 NERSC 上 `last.ckpt` 的 bytes，不能宣稱已驗證它現在的內容。

下一次測試直接使用指定的 `last.ckpt`，不再增加 SHA contract。Launcher 會檢查 checkpoint path、`global_step=1110`、必要 checkpoint keys、v5 report schema 與 policy 載入後的逐 tensor 相等性。從 live policy weights 啟動 `weights_only` 新實驗，重建 reward/reference pair、monitor、optimizer 與新 clocks。固定 classifier pretrained backbone 的來源另行記錄，不要意外拿 DGPO policy Body 當 classifier backbone。

初次啟動應使用新 output/W&B 路徑、`auto_resume_from_last: false`、`bootstrap_on_start: true`、`bootstrap_fail_closed: true`，不匯入 parent reward。若之後要處理 preemption，另以該 arm 自己的 checkpoint 完整 resume，避免不小心從 parent 重跑。

目前 `dgpo_omnifold_ztautau_10pct_old_method_hard_dropout015.yaml` 是另一個設計：full resume + 一次 Fourier refit，且 `refit_once_fail_closed: false`。它也設定 `[1]` 與 `warm_start_from_iteration_one: true`，不是這裡提出的 fully cold 實驗。

**Ensemble 與 memory 的具體定義**

優先測 3 repeats × 2 folds，而不是直接把 2 folds 改成 5 folds。5-fold 的每個 training event 仍只有一個 held-out model 產生 OOF increment，且每個成員的訓練比例從 50% 變成 80%，同時改了訓練資料量。3×2-fold 維持每個成員 50% fit population，讓每個 event 有三個沒用該 identity 擬合的 logits 可以平均。

在固定模型架構、資料與其他設定下，使用平均 **log-ratio/logit**，再形成 weight；不是相加六個未除以成員數的 logits，也不是平均 classifier probabilities 後假定比例估計等價。現有 `fit_evenet_residual_ratio` 已有 `crossfit_repeats`，OOF increment 用每 repeat 的 held-out logit 平均，獨立 validation 上則平均所有 repeat/fold models。

Fresh start 應包括 classifier-specific modules 和可訓練 Body adapters/input embeddings 回到指定 pretrained initialization、fresh optimizer，以及每輪重新由零計算 cumulative log weights。固定 pretrained backbone 的知識保留。重新初始化 seed 每 round/member 不同，但 outer split、identity fold seeds 與共同 evaluation noise 固定。不要重置 diffusion policy 本身。

排除全部 reward classifier inheritance 的設定片段：

```yaml
# 片段，不是可直接 launch 的完整 overlay。
dgpo:
  adaptive_omnifold:
    recalibration:
      crossfit_partition: identity
      crossfit_folds: 2
      crossfit_repeats: 3
      warm_start_iterations: []
      warm_start_from_iteration_one: false
      fit:
        min_steps_per_fold: 1000
        require_saturation: true
        restore_best: true
```

原先 optimizers 和 cumulative weights 已經每次重設，所以「avoid memory」要特別區分是否還繼承 trainable classifier weights、資料 identity，以及 model selection 所用的 validation history。

還有一個實作界線：`FrozenResidualRatioReward.forward` 現在會對每個 candidate 平均所有 checkpoint，沒有根據 event identity 選擇 held-out member。預設 `omnifold_train_shard` 也會使用 DGPO `train_shard`。因此，**fitter 的 OOF 權重不等於 DGPO online reward 已經 event-level OOF**。新 candidates 並不表示 conditioning event identity 沒見過。若要明確排除這類 memorization，應另測 disjoint classifier-fit/policy-update event populations，或實作 identity-routed online scoring；單改 repeats 或 cold-start YAML 不會做到這件事。

**建議的測試順序**

1. 固定 c4a91e07 last policy，生成一份共同 event/sample pool。獨立切出 classifier train、early-stop validation、最終 audit；重複 candidate 必須依 event identity 同組切分。Legacy classifier 與 nonlinear phi-H4 各自訓練到 plateau，使用同一 evaluation protocol。原始 run 的 50-update 舊 judge 與 2000-update H4 judge 不是容量純比較。
2. 在這份 pool 比較 H4 的 1×2 與 3×2，至少兩個整體 training seeds。先看第1個 increment，再檢查第2/3個增量；允許診斷上限時未 closure，不能把達到 iteration cap 標為成功。記錄完整 fit/OOF/audit BCE、logit disagreement、ESS、top-1% mass 和 weighted conditional physics。此階段不更新 diffusion。
3. 只有穩定的候選才進入從同一 snapshot 開始的 50-step DGPO pilot，再考慮延伸到100。Arm A=H4 cold 1×2；B=H4 cold 3×2；C=H4 3×2，只繼承前一輪同 fold iteration1、later iterations cold。A/B 測 ensemble，B/C 測跨輪 memory。所有 arm 起始 classifiers 都是 cold。使用相同 minimum fit budget/validation protocol，並另外報 GPU-hours；不要將 cold1000 與 warm10-epoch 的速度差混成初始化效果。
4. Pilot 保留 soft velocity-MSE trust coefficient=1，關閉 adaptive hard-trust boundary。舊 v26 lineage 的 hard boundary 從未縮放 accepted step，而 c4a91e07 的 refit shock 發生在 full-LR transition；因此先不讓 hard boundary 把 Fourier update 壓到近零。Policy LR 暫以 parent 1e-4 的四分之一即2.5e-5 作保守起點，所有 policy param groups 同比例縮放；每次 reward install 20-step warmup（0.1→1），沿用舊 v26 穩定 lineage 的 transition protocol。這是待驗證設定，不是已證實的 optimum。每10 policy updates 看一次 physics；各 arm 使用完全相同的 controller/cadence，防止把 refit 次數差當作 ensemble 效果。
5. Tempering 先在固定 policy 上配對比較0.75與0.5，再為所有 pilot arms 選同一值。預先設 ESS/N≥0.2 作探索性拒絕門檻，並記錄各 channel/truth-bin 的 ESS 和 tail mass；門檻不是物理定律，不能為追求高 ESS 自動把 reward 壓成常數。既有 ESS0.3 試驗也沒有證明這個 guard 足以改善 physics。不要只為了安裝 reward 而放寬 closure gate。

3×2 相對1×2，每個 residual 有六個而非兩個 classifier fits；也增加 online reward scoring 成本。降低方差是要測的結果，不是必然有3倍有效樣本，因為成員誤差仍相關。每輪 cold start 的額外時間尤其要納入比較。

現有 `diagnose_residual_weights.py` preflight 固定要求 fullfit、iteration1 inheritance、single-pool protocol；不能只把 diagnostic YAML 的 checkpoint 路徑改掉，就當成上述 cold-H4 測試。需要新增獨立 protocol/launcher 後再執行；本次沒有修改或啟動它。

**Response matrix 必須成為直接驗證項目**

Correlation 或若干 joint distribution 的改善，不保證每個 truth bin 的 conditional migration 改善。以 `R[i,j]=P(reco bin i | truth bin j)` 為例，固定 policy/reconstruction 下，若 reweighting 在某個 truth column 內近似常數 `w[j]`，column normalization 會使 `w[j] N[i,j] / sum_i w[j] N[i,j] = R[i,j]`。總體分布可以變，該 conditional response 卻不變。這是固定樣本 reweighting 的例子；真正 DGPO policy updates 可以改變重建值，不能把上式當成 DGPO 永遠不能改善 response 的論證。

每個選定 observable/channel 使用相同 event identities、binning、selection、MC weights、candidate estimator、sampling seeds。主要看 truth-normalized matrix 的每-column diagonal/near-diagonal fraction、mean/RMS migration、bias/resolution；同時保留 efficiency、fake/miss、under/overflow。全體平均需要固定 reference truth-bin/channel 權重，避免改變 truth occupancy 就產生表面進步。以 event 為單位做 paired bootstrap；同一 event 的 K candidates 不是 K 個獨立觀測。

Repo 的 `extract_response_matrix_summary.py` 已有 diagonal fraction、near-diagonal fraction、mean/RMS bin offset 與 `--normalize truth` 圖。它目前的 aggregate metrics 不等於完整逐 truth-bin calibration/coverage；後者要另算。最終還要用相同 unfolding 設定比較 closure bias、uncertainty 與 coverage，不能只追求更對角：縮窄到錯誤中心也可能讓圖看起來更集中。

必須先釐清目前 response 用 single posterior sample、sample mean，或其他 event estimator。對完美 posterior 而言，令 truth 與一個 generated draw 在條件x下獨立，則單一 sample 的條件 MSE 是 `2 Var(O|x)`，K 個 iid samples 的 observable 平均是 `(1+1/K) Var(O|x)`。因此額外 sampling noise 可以使完美 posterior 的 response 仍然寬；這是數學上的條件式例子，不是此次模型已校準的證據。若要測 K=8/16 的 posterior mean，應平均最終 physics observable `O(z,x)`，不是直接平均環狀角度或先平均 tau vectors 再套非線性 observable。這會改變 estimator，需要另建 response 和重新驗證 bias/coverage，不能拿它與 single-sample arm 混作 classifier-only 比較。

如需補強 classifier 對目標的敏感度，下一個獨立 feature ablation 才加入實際 response observable 的 candidate-dependent features 和可觀測 event/channel context；truth/generated 兩類必須採同一計算規則，generated branch 不可讀取該 event 的隱藏 truth。不要把條件不足直接歸因於 phi Fourier harmonics 不夠。

最後，統一 inference protocol：W&B DGPO validation 使用 live policy、20 DDIM steps；`util/predict_evenet_from_raw_parquet.py` 的 CLI `--num-steps` 預設是200，EMA 選擇則依 config，能用 `--disable-ema` 固定 live weights。作比較時應明確指定 `--num-steps 20 --disable-ema`，並確認相同 sampler。這是需核對的混淆來源，尚無證據表示使用者之前的 response evaluation 用錯設定。

Generative unfolding 文獻同樣區分 distribution matching 與正確 conditional probabilities；參見 [Generative Unfolding with Distribution Mapping](https://arxiv.org/abs/2411.02495) 與 [Event-by-event comparison](https://link.springer.com/article/10.1140/epjc/s10052-024-13136-3)。這些提供測試設計的背景，並不能證明本專案的 ensemble 或任何新超參數會成功。
