# Learned weak second stage from pure-strong10k

Source: truth_three_arm_v1/strong_no_kl/state.pt, RAW step10000, new truth-trained
lineage. No transport, whitening, original Gaussian-teacher checkpoint, EMA, or
classifier reuse. Fit one plain/no-Fourier classifier on original fixed truth rows
(32768/8192/16384), with newly generated fixed negatives from this source.
Seed117, BCE selection, patience20/min_delta1e-4, AdamW3e-4; no fit step cap.
Same architecture, Gaussian-CDF features and head standardization as prior plain fit.

Freeze selected weak; run10000 second-stage DGPO updates, fresh AdamW1e-4/WD.001,
batch64,K8,M4,DDIM20, unchanged unscaled LOO objective. Add velocity-MSE coefficient1
against the second-stage starting checkpoint, NOT the original pretrained model.
This reset is intentional; it may resist repair of the already distorted source.
Policy seed117, monitor330017, independent endpoint332017, no refits.

Primary: all three low-order errors (max residual mean, variance error and pair
covariance) lower than stage1, while56-moment RMSE remains below original pretrained
baseline. Report point criterion separately from paired moment-MSE confidence
intervals vs both starting states; no claim of full closure from relative improvement.
Own fixed reward absorption, within-K ESS, informative candidate fraction and gradient
conflict monitored. Structure every250; full optimizer/RNG checkpoint every1000.
All checkpoints immutable inputs; outputs under pure_strong10k_weak_repair_v1.

Risk: large source residuals saturate Gaussian-CDF features. High AUC can coexist
with weak within-condition candidate discrimination. Do not silently change features
to force a positive result. Single seed, fixed-reference toy; no production claims.

Run:
python -u -m experiments.dgpo_toy.weak_repair --output artifacts/dgpo_toy/pure_strong10k_weak_repair_v1
