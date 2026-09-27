# User preferences

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
