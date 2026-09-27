# H4 pair-token classifier stability (`h4pair01`)

## Question
Can the same nine angular features learn reliably as a decoder token without
an independent Fourier encoder, late fusion, or fitted output standardizer?
The user selected the step-1110 policy and the existing clean validation truth
pool. This is classifier discrimination, not policy closure.

## Frozen contract
- User: weights-only c4a91e07 `dgpo-epoch=110-next_ep=111-step=1110.ckpt`.
- User: `diffusion_val_20pct_seed42_stic_filtered_test1/val` truth pool.
  The 10% label describes the checkpoint lineage, not this truth directory.
- h4ratio1 control: one cold fit, identity-disjoint fit/early-stop/final-test
  partitions, K=1, original clean data/normalizers and 16 GPU workers.
- h4ratio1 control: constant head LR 2e-4, adapter/decoder LR 5e-5,
  last-PET-block LR 1e-5, AdamW weight decay .001, clipping 5, dropout .25.
- h4ratio1 control: 1000 minimum / 3000 maximum updates, ten effective epochs
  BCE patience with min_delta .001, restore every strict best validation BCE.
- User: same sin/cos(n delta_phi), n=1..4, plus opening cosine. No theta-pair
  extension in this comparison. No reward fit or policy update.

## Architecture
A learned linear projection maps nine bounded pair features to one 128-wide
token. Append it after the two candidate-token projections, before the first
decoder self-attention. All three tokens share the existing self-attention,
visible-memory cross-attention, AdaLN/FFN and output LayerNorm. Flatten all
three outputs to the zero-initialized scalar logit head. One decoder block,
four heads, unchanged. No extra pair positional embedding is required: its
separate input projection and fixed readout slot identify its role.

Delete the topology encoder, its fixed output standardizer, and late-fusion
MLP in this opt-in arm. Keep all legacy defaults and checkpoint paths intact.
The architecture changes capacity and RNG consumption; a same-seed historical
comparison is a pilot, not a replicated causal estimate of stability.

## Measurements and decision rule
Primary: mean validation BCE at shared logged steps in [200,300], compared
with the downloaded h4scls1-retry1 trajectory (0.53036789). Promising early
learning requires no worse than +.01 BCE at this endpoint. This practical
margin is a pilot threshold, not a significance claim.
Secondary: BCE at shared steps 200/300/800/1000, full train/validation curve,
preclip norms, clipping fraction, nonfinite failures, final test BCE/AUC and
actual fit/best steps. Compare h4ratio1 over shared realized training steps.
Do not claim improved stability from the early BCE endpoint alone: report
clipping frequency and validation rebounds (> .02 above preceding best),
including all updates and budget limits. Across-seed stability remains open.

Export the restored best classifier and disjoint held-out scores for raw and
fixed-.75 ESS, top1% and maximum weight mass, mean ratio and topology diagnostics.
Coarse cosine JSD is not a success criterion; it hides the endpoint spike.
No clipping, temperature or checkpoint choice on final test. If the token arm
learns poorly, report unresolved/negative for this architecture; do not add
normalization, depth or theta features in the same run.

## Required result table
| Arm | Updates / best step | BCE @200 / @300 | Mean BCE 200–300 | Test BCE / AUC | Clip fraction / rebounds | ESS / top1% mass |
|---|---|---|---|---|---|---|
| h4pair01 | pending | pending | pending | pending | pending | pending |

## W&B and run
ID `h4pair01`, group `H4 feature fusion`, display name
`Can pair tokens stabilize learning? | H4 decoder input | step-1110 classifier`.
Use classifier fit-step axes, not policy global_step (always zero).

```bash
shifter python3 scripts/train_h4_pair_token.py --check-only
shifter python3 scripts/train_h4_pair_token.py
```
`--validate-only` checks the resolved configuration on a machine without NERSC
files. `--check-only` additionally requires the source checkpoint, truth data,
clean-data manifest and unused output directory. Launch saves resolved config
exclusively. A retry requires a new `--run-suffix` for isolated outputs/W&B ID.

## Musk gate
- Link tested: classifier feature fusion/optimization. Owner: user's request.
- Requirement owners are listed in the frozen contract above.
- Deleted: topology encoder, fixed standardizer, fusion MLP, frozen Fourier
  ridge-path exports. Retain branch measurements for the new pair projection.
- Fastest falsifier: one cold fit with first-300-step learning screen. Saved
  frozen-head replay cannot test a token inside decoder attention.
- Time to first screen: 300 optimizer updates plus pool generation. Wall time
  is unmeasured for this architecture; no invented GPU-hour estimate.
- Kill criterion: fail on invalid inputs/nonfinite training; maximum 3000
  updates. Do not kill at early near-chance AUC, given prior delayed learning.
- One run, no additional allocation requested automatically. Expected cost:
  16 * elapsed hours / one potentially resolved hypothesis.

## Session record
Limiting factor: classifier feature fusion. Deliverable: tested implementation
and runnable experiment. Time box: about one hour for local implementation.
Existing registry backlog is not re-audited (last recorded nine); user explicitly
prioritized this architecture. No new scientific result is claimed here.

## Implementation validation (2026-09-21)
Implemented; no NERSC training run launched. New architecture tests: 15 passed.
The broader selected suite passed 272 tests and 66 subtests with one new test
assertion failure (the replay helper returns a tuple). After correcting that
test, all 15 new tests passed, covering the previously failing case. Thus all
273 selected cases passed across these runs. Distributed tests ran outside
the sandbox because local shared-memory/socket restrictions prevented them.

One unrelated existing test was excluded from the selected rerun:
`test_best_point_restart_opts_inherited_reference_in_at_age_zero`, which fails
in the untouched best_decay trust initialization logic. It also failed in the
initial broad run. No trust behavior was changed to repair it here.

Verified: third token enters attention; open-gate pair perturbations affect
physical candidate streams; pair projection receives gradients and updates;
B-by-K candidates match separate evaluations; PEFT restore reproduces logits;
legacy banks omit the new false flag; builder/bootstrap/audit/replay routes
preserve the option; conflicting branches fail; branch diagnostics remain
complete. Resolved launcher config and whitespace checks pass.

NERSC SSH returned `Permission denied` during the read-only connectivity check.
Files remain local. Refresh NERSC authentication before upload and launch.

Session close: architecture and checkpoint wiring were verified before GPU
work. Next session should record measured runtime and full BCE trajectory.
Registry debt was not re-audited; no new executed run was added by this session.
