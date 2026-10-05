# Attention reward DGPO continuation

Launch after syncing updated code, in the user's existing 16-GPU allocation:

```bash
shifter python3 -u scripts/resume_tau_attention_dgpo.py config/dgpo_tau_attention_1780.yaml
```

Initial source is the pinned full-state step1780 checkpoint from `5a7bf14b`.
This is full resume: actor, optimizer, scheduler and epoch clocks are retained.
Before any actor update, cold-fit the three-block FiLM + 64-wide four-head
candidate-query cross-attention classifier on current-policy K1 negatives.
Use the existing filtered 416701-event training panel; its existing internal
split is 354488 fitting / 62213 best-BCE selection. The separate 119002-event
validation panel is never used to refit/select. No new Cij input or loss.

At installation, replace—not multiply—the old ratio; recenter the velocity-MSE
reference to the current actor. Keep coefficient 1. Save the new reward and
reference immediately, before the first update. Refit every five completed
epochs since the last install (first at completed epoch 183 / zero-based 182
when resuming at start_epoch=178). Periodic refits are fresh, not warm starts.

Preserve inherited fit settings: 16 GPUs, 1024 paired events/GPU, 250 epochs,
1000 minimum steps, patience25, best internal-validation BCE, bound30.
Preserve actor LR/schedule, no-Fourier architecture and raw—not EMA—weights.
Validation/audit cadence remains the inherited ten epochs. Fresh audits also
use the attention architecture, with the existing unbounded audit objective.

Output is isolated under `dgpo_tau_attention_1780`; production source files
are not overwritten. The same launch command resumes this output's last.ckpt
after interruptions and creates a new W&B run. A saved attention head prevents
the initial migration refit from being repeated on every restart. Refit cadence
uses the saved last_refit_epoch, not process startup time.

W&B reports refit fit/selection, reward round, Cij, per-component closure,
and global/within-condition reweighting probes. Head architecture is stored
in each best.pt and the full DGPO reward stack. Primary endpoint is actual
unweighted generated Cij improvement, not only reweighted candidates or AUC.

Local tests cover configuration, five-epoch cadence and attention forward,
masking, gradients and serialization. Full distributed integration is not
executed locally; requires the user's production environment and allocation.
