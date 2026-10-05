# User preferences

- The ultimate physics analysis is tau-pair spin correlation / entanglement
  (confirmed 2026-09-29). Prioritize downstream polarimeter, spin-correlation
  matrix and entanglement-observable closure when judging physics usefulness.
  Angular marginals, lab-frame opening/acoplanarity and narrow joint peaks are
  supporting diagnostics, not substitutes for spin closure. Do not identify
  momentum opening angles with spin correlations or direction-only outputs
  with fully reconstructed tau rest frames. Retain predeclared experiment
  endpoints and H4 audits, but distinguish them from this ultimate objective.

- Prepare real-case diagnostics and experiments for 16 GPUs by default, including
  inference-only evaluations. Do not reduce to one GPU unless the user asks.

- Use the existing filtered datasets for all new/repeated real-case experiments:
  train `/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`
  (416,701 events), validation
  `/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`
  (119,002 events). Check the completed filter manifests; never silently fall
  back to the raw datasets. Preserve the pinned checkpoint's normalization.
  A different population/fraction requires a separately verified filtered input;
  do not silently substitute the 10% population into a differently sized study.

- The user submits all compute jobs personally. Prepare and validate code,
  experiment configurations, and launch commands, but do not submit jobs,
  request allocations, or start remote training. Requests to implement/train
  an experiment do not override this standing preference unless the user
  explicitly authorizes the assistant to submit that specific job.
- Read-only remote inspection is permitted. Cancel a job when explicitly
  requested by the user.
- Exclude toy-model code and outputs from uploads/synchronization to NERSC
  unless the user explicitly requests them. In particular, exclude
  `experiments/dgpo_toy/` and `artifacts/dgpo_toy/`. For repository-root rsync
  uploads, use `--exclude-from=NERSC/upload-excludes.txt` in addition to any
  existing exclusions. Do not delete previously uploaded files implicitly.
- Also exclude `artifacts/c4a91e07_review/` and
  `artifacts/classifier_signal_comparison/` (including all saved classifier
  checkpoints) from uploads. Update the existing remote `ml_pipeline` for
  script runs rather than creating separate remote code copies.
