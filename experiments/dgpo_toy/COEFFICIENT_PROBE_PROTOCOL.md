# Old-baseline coefficient0.1 probe

Only new training arm: lambda=.1,10000 steps, seed17, frozen old joint3 classifier
and raw Gaussian-teacher initial from low_ess_joint3_v1/reward.pt. Reference stays
at that same initial. No reward rescaling, classifier fits, feature changes, or
optimizer changes. Reuse completed lambda0 and lambda1 controls. AdamW1e-4/WD.001,
batch64,K8,M4,DDIM20; train3017/monitor90017/endpoint94017 match lambda1.

Primary: negative paired56-moment MSE-change upper95CI vs initial AND max residual
mean<.05,varianceerror<.10,offdiagonal covariance<.05. Thresholds operational toy
gates, not physical tolerances. Report reward separately, never as structural proof.
No post-hoc checkpoint selection; endpoint10000 fixed. Finite failure preserves
latest1000-step state; no automatic coefficient adjustment. Single training seed.

After training automatically evaluate all three endpoints plus initial using fresh
calibration32768/seed340017 and paired evaluation16384/seed350017. Fit pooled
residual Gaussian-CDF transport separately per model on calibration only. Report
raw and repaired moment changes; repair is diagnostic, never applied to training.
Bootstrap uncertainty conditional on fitted map. Record whether raw improvement
survives repair; covariance/conditional-bin errors remain visible.

Command: python -u -m experiments.dgpo_toy.coefficient_probe --output artifacts/dgpo_toy/velocity_kl01_joint3_10k_v1

Training report.json completes before repair_diagnostic.json; distinguish both states.
Training retains original velocity_kl runner and exact first-update match check.
