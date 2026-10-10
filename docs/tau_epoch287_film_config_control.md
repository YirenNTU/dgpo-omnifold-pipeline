# Original pretrain epoch287: configurable FiLM

Use the original 10% pretrain checkpoint (stored epoch286/global_step18655),
raw weights and its pinned normalization. This explicitly requested historical
configuration does not migrate the checkpoint to the later MET/two-leg adapter.
Original learned FiLM parameters stay present and strictly load in every mode.

For the full runtime, edit
`config/eval_diffusion_epoch287_film_control.yaml`:

```yaml
network:
  VisibleConditioning:
    diffusion_enabled: true
    diffusion_film_gains:
      attention_gamma: 1.0
      attention_beta: 1.0
      ffn_gamma: 0.0
      ffn_beta: 0.0
```

The shipped default has all four gains1 (exact original behavior). The above
example disables FFN-FiLM only; the FFN itself remains. Use0.6 for weaker
FiLM, set only gamma or beta0 to isolate that component, set attention
coefficients0 to disable attention-FiLM, or all four0 to disable both.
The identity term in `(1+gain_gamma*gamma)*LN(x)+gain_beta*beta` is retained.
Time/global/process-label conditioning stays unchanged. Finite nonnegative
numbers are accepted; missing coefficients default to1; unknown keys, bools,
strings and NaN/Inf are rejected. No new model parameters/buffers are added.
Gains apply once at the generation block input, after the FiLM coefficients
are assembled. Existing binary gamma/FFN disable flags take precedence.

For external evaluators accepting a standalone network YAML, use
`config/network_diffusion_epoch287_film_control.yaml` and edit the same
`VisibleConditioning.diffusion_film_gains` section (no outer `network` key).
The full runtime and standalone export are independent files; edit the file
that your evaluator actually reads. The evaluator must import the updated
`evenet_dgpo/evenet` implementation, or these new runtime controls will not
be understood. Keep `diffusion_enabled: true` even when all gains are zero:
setting it false removes the learned FiLM module and its weights.

The full runtime pins the original raw snapshot at
`/pscratch/sd/y/yiren/Ztautau/tau_dgpo_spin_angles/within-event-reward-epoch287-refit10-val5-run-02/source.ckpt`,
with its matching normalization. No training or allocation is launched.
CPU strict-load/function check on NERSC:

```bash
shifter python3 -u scripts/check_tau_epoch287_film_control.py \
  config/eval_diffusion_epoch287_film_control.yaml
```

This reads two cached real validation events, runs DDIM20, checks all575model
tensors, then compares configured gains to an independent direct modulation
hook. It verifies control and loading, not physics precision. Existing five
inactive FAMO training states are reported separately from model tensors.
Local validation also checks default1, FFNoff, FFN0.6 and attentionoff using
the exact downloaded raw epoch287 checkpoint. Nonbinary controls are tested
against accidental double application. No optimizer update is made.
