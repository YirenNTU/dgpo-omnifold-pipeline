# Pure-reward／低 ESS toy：本機結果，2026-09-21

## 結論先行

**已實作並跑完三個 seed：低 ESS、權重集中、早期 reward 學得慢確實出現；
但未通過事前定義的600-step強烈停滯門檻。** 不改門檻、不繼續縮窄分布來製造成功。

這次不含任何 additive KL／reference-trust penalty。保留原 DGPO gate 內的
frozen-reference MSE 比較；這是原 main loss 的一部分，不是偷偷加回 penalty。
所有 production 檔案都未修改。

問題依使用者修正為「pure reward 是否吸收得動」，不以 truth KL、AUC 或 physics
closure 偷換 endpoint。`elon-musk-workflow` 要求事前定義判定與有效對照；
`omnifold-guided-learning` 提醒 global ratio ESS 不能代替 within-event 訊號與 gradient。

## 最小設計

一個 context c 均勻分布在圓上；一個參數的 Gaussian v-predictor 經 DDIM20 產生 z。
將觀測量定義為 x=c、y=(c+Phi(z/sigma_ref)-0.5) mod1。
不論 z 的分布怎麼變，x、y 各自都精確 uniform。

Truth residual 是20% reference Gaussian＋80%窄 Gaussian，形成 joint diagonal ridge。
寬版的 spike sigma/reference sigma=0.25，窄版=0.0002。
固定 reward=精確 log(p/q_ref)，不 fit classifier、不 temper、不 clip、不 refit。
模型已知有用的 residual 座標；不測它能否自己從 neural representation 發現這個座標。

兩版各跑三種方法，seeds17/29/43，600 updates、batch256 groups、K8、M8、DDIM20，
AdamW LR0.03／decay0.001／gradient clip1：

- **DGPO**：直接使用 repo 的 LOO advantage 與 detached nonlinear gate main loss。
- **Score-MC**：同樣 detached candidates／LOO，但使用精確 endpoint log-density
  的 score-function gradient。是不同 estimator，不宣稱等於 DGPO surrogate。
- **Population**：數值積分 expected reward 的梯度；同模型與 optimizer，
  但有額外積分能力，不是 compute-matched stochastic control。

Protocol：[NARROW_PROTOCOL.md](NARROW_PROTOCOL.md)。
實作：[narrow.py](narrow.py)、[narrow_probe.py](narrow_probe.py)。

## 是否真的低 ESS？是

| 初始集中度 | 寬版 | 窄版 |
|---|---:|---:|
| 解析 population ESS/N | 45.47% | **0.04419%** |
| 23927獨立 MC 樣本 ESS/N，三 seed範圍 | — | 0.04344–0.04806% |
| top1% ratio weight mass，三 seed範圍 | — | **78.4–84.8%** |
| 最大單點 weight mass | — | 12.8–18.2% |

過去 h4ratio1 約0.038% ESS/N、top1% mass82.6%。目前 toy 涵蓋相近的集中程度，
但沒有使用真實 classifier，也不能藉此宣稱相同因果機制。Toy ratio 是精確正規化的，
不同於實際 ratio-health 的 normalization 警訊。

## Reward 有沒有學不動？早期很慢，後來仍會改善

下表為已取得的 available reward gain：

`gain_fraction=(E_q[r]-E_q0[r])/(max_z r(z)-E_q0[r])`。

上界是固定 reward 的解析最大值，不是 truth distribution 的 reward；沒有 KL 時
不把集中到高 reward 區當成 density closure。

| 方法 | step100 | step300（次要讀數） | step600（事前主 endpoint） |
|---|---:|---:|---:|
| 寬 DGPO | 71.26–71.59% | 92.75–92.83% | **96.63–96.67%** |
| 窄 DGPO | **0.092–0.105%** | **2.45–2.91%** | **33.59–37.25%** |
| 窄 Score-MC | 0.155–0.169% | 99.026–99.042% | **99.497%** |
| 窄 Population | 0.361% | 99.082% | **99.496%** |

窄 DGPO 的 raw expected reward：

- 初始：-1.604689。
- step100：-1.595577至-1.594253，確實幾乎不動。
- step300：-1.362155至-1.316799。
- step600：+1.719932至+2.082267，已有明顯改善，不是永久停滯。

Population 三個相同 endpoint 是確定性的控制，不是三次獨立 optimizer replication。
訓練樣本與獨立 MC evaluation 分開；主指標用高精度積分，避免窄區沒有被 eval 抽到。

**正式 decision=`not_reproduced`**：事前要求窄 DGPO gain不超過寬 DGPO的25%，
也不超過窄 Population的25%；實際約35–39%及34–37%，三 seed都不符。
不因 step300 看起來更像失敗，就把主 endpoint 從600移到300。
因此沒有觸發 protocol 中「成功重現後」的 batch8192 training check。

![Matched broad and narrow reward trajectories](../../artifacts/dgpo_toy/narrow_v1/curves.png)

## 訊號集中在哪裡？不能只看 global ESS

窄 DGPO 前50 steps，三 seed的時間平均：

- 只有約0.94–1.04%的 K8 groups，其 reward range大於1e-3。
- 約0.45–0.49%的 groups抽到至少一個 |z|<3*spike_sigma 的 candidate。
- **mean within-K weight ESS/N卻約99.35–99.44%**：大部分組內候選同樣不好，
  權重看似平均，不代表有足夠的選擇訊號。
- 以每批 sum|gradient| 加權，top1% groups占約94.6–96.6%的 absolute gradient。
  這只是 scalar toy 的梯度貢獻集中度，不是 production event influence。
- gate mean約0.5，沒有整體 sigmoid saturation。

CSV仍保留極小數值梯度。少數沒有可見 reward差異的 batch只有 LOO 浮點誤差，
其 ESS比例沒有實質意義；解讀 concentration時要同看 `gradient_contribution_total`。
不可把圖中這些近零梯度的高 ESS尖點當成訊號改善。

此外，連 Population控制前100 steps也很慢，因此最初平坦的一段不能全歸因
「沒有抽到好 candidate」；從初始寬分布移到窄區的參數距離／優化動力學也在作用。

## 大樣本、零更新的 gradient 檢查

這是完成 screen後事前列明的 exploratory follow-up，不改原判定。
凍結 seed17的 DGPO step0/300/600，四份各8192 groups，兩種 estimator共享
同一批候選／noise。比較其平均梯度與同一位置的 Population reward梯度。

| Anchor | DGPO gradient / Population gradient | Score-MC / Population |
|---|---:|---:|
| step0 | 0.3023 | 1.1195 |
| step300 | **0.001170** | **0.9910** |
| step600 | **0.00002382** | **0.9998** |

step300 DGPO mean gradient0.0003033、MC SE0.0000090；Score-MC mean0.25682、
MC SE0.00060。所有 anchor、兩種方法在四份 panel的方向都與 Population一致。
SE描述這四份 toy MC panels，不是 production uncertainty。

**支持的候選機制：**在這個窄 joint toy裡，DGPO main的平均梯度相對於
expected reward gradient會隨 policy位置大幅衰減；不只是單次少量抽樣碰巧沒有訊號。
Score-MC在相同有限候選預算能學好，排除「這批候選完全沒有足夠訊號」作為此toy的充分解釋。

**不能推論：**梯度小幾萬倍就等於效率差幾萬倍。AdamW會重縮放梯度，且其歷史二階矩
會影響實際更新。這個比值不是可直接套用的 LR倍率，也尚未區分 velocity-MSE、
time weighting、finite-DDIM geometry各自的貢獻。这里只有一個參數，沒有測高維方向旋轉。

## 有效性與限制

- 22項測試通過：包括原toy、density normalization／ESS、uniform marginals、
  quadrature梯度與解析score identity、per-group梯度總和對autograd、DDIM witness、
  deterministic replay與evaluation RNG隔離。測試不要求發生預期失敗。
- Quadrature256→512的最大 reward誤差2.5e-14，檢查點最大梯度相對差4.6e-13。
- 第一個 smoke在窄版訓練前因DDIM noise floor下的witness不可達而停止；修正為
  可達witness，保留sampler／訓練／門檻不變。紀錄於
  [incomplete smoke](../../artifacts/dgpo_toy/narrow_smoke_v1/FAILURE.md)。
- Width同時改變 ESS、reward形狀、最佳區域寬度和需要移動的參數距離，
  **不是只改ESS的因果實驗**。真正 matched estimator comparison是在同一個width內。
- 只用一個 analytic diffusion參數與已知residual座標；不代表真實conditional neural
  generator的可達性、H4學到的高階結構或實際16-GPU batch。
- 無learned critic、不存在 ratio估計偏差；max reward也不等於physics成功。

## 重跑與資料

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -m pytest -q experiments/dgpo_toy
/opt/miniconda3/envs/MyEve/bin/python experiments/dgpo_toy/narrow.py --output artifacts/dgpo_toy/narrow_v1
/opt/miniconda3/envs/MyEve/bin/python experiments/dgpo_toy/narrow_probe.py artifacts/dgpo_toy/narrow_v1 --output artifacts/dgpo_toy/narrow_gradient_probe_v1.json
```

[Report](../../artifacts/dgpo_toy/narrow_v1/report.json) ·
[每步CSV](../../artifacts/dgpo_toy/narrow_v1/history.csv) ·
[Gradient panels](../../artifacts/dgpo_toy/narrow_gradient_probe_v1.json)。
命令只覆寫指定輸出中的report／CSV／plot，不清除其他檔案。不需GPU／W&B／NERSC。

## Research round close
gi
- **Outcome：**符合低ESS與早期遲滯；未達600-step強烈停滯定義。
- **Evidence：**三 seed；同樣候選預算Score-MC成功；大panel梯度顯示位置依賴的尺度衰減。
- **Decision：**支持 estimator／diffusion geometry作為此toy的效率瓶頸；
  production同因、持續停滯與ESS單因解釋仍unresolved。不改production。
- **Learning index：**三seed screen8.41s＋probe0.27s，約8.68s／兩項判斷更新=
  4.34s／更新。只計訓練與量測，不含開發、啟動、畫圖、smoke和測試。
- **Deleted：**KL、learned classifiers、refit、特徵工程、外部compute；未因未達門檻而繼續調窄width。
- **Next limiting factor：**哪個main estimator成分造成 policy-dependent attenuation，
  以及它是否能解釋actual AdamW displacement，而不只gradient norm。
- **Better next round：**先在固定toy anchors做單一成分干預並比actual reward gain，
  保留這次600-step判定，不移動門檻或把改善後的toy直接當production解法。
