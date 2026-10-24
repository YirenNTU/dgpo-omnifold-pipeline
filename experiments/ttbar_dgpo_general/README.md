# TTbar General70 + reconstructed-top oracle30 DGPO

Prepared for manual launch only. No allocation, training, or classifier refit is
started by `--check`. This adapter uses the native EveNet DGPO trainer and its
optimized DDP accumulation, rather than copying a second trainer.

## Pinned experiment

- Initial policy: the successful full-MG5 visible-FiLM **epoch115** checkpoint
  copied into classifier run02's `policy.ckpt`. Its complete saved `policy.yaml`
  supplies the architecture and normalization. No architecture transplant is
  performed during DGPO. Assignment loss and EMA remain disabled.
- Initial classifier: frozen-feature General run02, best epoch172, validation
  BCE 0.63933656; its original fit reached the plateau criterion at epoch222.
- **16 GPUs, 512 events/GPU, K8**. Global rollout:8192 events/65536 candidates.
  Two candidate chains are batched per generation forward:1024 candidates/GPU.
  This is an inference memory cap, not a reduction of the optimizer event batch.
- Full MG5 training; seed42. Current validation remains developmental because
  MG5 test was already used for diffusion checkpoint selection.
- AdamW head LR1e-5, backbone LR1e-6, weight decay0.01; beta1; eight independent
  policy-evaluation timesteps per optimizer update, t in[0,0.7], DDIM20.
  These are explicit initial ttbar settings, not an assertion of optimality.
- Unscaled leave-one-out advantage, no advantage clipping. Native reference
  trust coefficient1 (native loss is0.5×velocity MSE), no latent-SWD/CPO/PID.
- Stop at15000 updates or200 full data epochs, whichever comes first. This is
  a bounded preparation, not a claim that the policy has saturated.

## Reward contract

For each event's eight joint neutrino candidates:

1. The fixed epoch115 feature model produces clean t0 pre-velocity tokens.
   The General head also receives the unchanged observed inputs and ordered
   neutrino geometry. MC truth never enters the General score.
2. `q = softmax(General logits / 1.0)` across the eight candidates.
3. Oracle candidate: reco b+reco lepton+predicted neutrino, both ordered legs.
   Oracle reference: the SAME visible objects+MC neutrino. Minimize the mean
   squared p4 difference over both legs and E,px,py,pz. The common visible p4
   cancels exactly; the implementation uses that identity, not another target.
4. Fixed training-only top-p4 RMS scales are
   `[402.3031118, 210.9381482, 210.9381482, 210.9381482]`GeV: one energy scale,
   one shared xyz scale. They are never re-estimated on validation or refits.
5. `w = 0.7*q + 0.3*one_hot(nearest)`; each event sums to1. Compute `log(w)`
   with log-softmax/logaddexp, then `A_i = log(w_i)-mean_{j!=i}log(w_j)`.
   This is not an additive combination of two scalar rewards. Thirty percent
   weight is not a promise of thirty percent gradient contribution.

The oracle needs training truth. Inference and raw-policy physics evaluation
use the diffusion's unweighted predictions.

## Periodic refits and efficiency

Every500 optimizer updates, freeze the current policy to generate one new
negative/event for the full MG5 training population and classifier-validation
selection. The pre-existing positives and conditions are reused only after
exact event-ID and target equality checks; feature backbone and normalization
remain fixed. All ranks generate their disjoint source row chunks. There is no
nested Ray trainer or new cluster.

Warm-start the General head, reset its AdamW, fit BCE with batch2048/GPU,
LR1e-4→1e-5 cosine over1500 epochs, dropout0.1, weight decay0.005. Stop after50
stale validation epochs and at least1000updates (cap3000epochs). Failure to reach
plateau stops the run instead of silently installing an unfinished head. The
best validation-BCE head replaces the prior head; this is a fresh density-ratio
fit to the current policy, not cumulative multiplication of old head ratios.

After installation, recenter the frozen velocity reference to the policy that
produced the negatives. The alpha remains0.3. Refits preserve the policy RNG
stream, use the existing distributed group, and save their own interrupted-fit
state. Cache policy digests prevent reusing negatives from different weights.

Training uses the tested native `no_sync` forward+backward accumulation and
one gradient reduction on the last substep, collective finite-loss checks, and
no unnecessary CPO parameter snapshots. Reward scoring stays in torch on GPU.
The source SHA checks require the current efficiency update. Nonfinite rollout
values cause failure; they are not silently replaced with zero in this arm.

## Checkpoints and validation

All output is under:
`/pscratch/sd/y/yiren/Preference_Optimization/ttbar-dgpo-general70-top-oracle30-full-mg5-run-01`

`checkpoints/last.ckpt` is atomic and saved every10updates/end of epoch.
Named `step-NNNNNN.ckpt` snapshots are retained every50updates and at validation.
Each checkpoint contains optimizer state, installed head, fixed statistics and
source hashes, refit step/round, and paired velocity reference. Explicit resume
must match the run contract. Native mid-epoch resume replays that data epoch;
it is not bitwise continuation of Ray's shuffle iterator/RNG.

`validation/step-NNNNNN/` contains full source-ID-matched K8 caches, before
training and every full epoch (roughly85updates). Validation directly iterates
stable parquet chunks so it does not truncate at the shortest Ray shard. The
same candidate seeds are reused across checkpoints. It logs raw-policy pT/eta/
phi and top-mass TV, conditional spread (circular for phi), ESS and reward.
`val/winrate_evaluated=0` explicitly marks the native compatibility winrate field
as unevaluated; do not interpret its placeholder as a physical result.

The cache preserves the existing classifier/physics event partitions. Full
saved candidates support local RooUnfoldSvd without new generation. Use
`evaluation-config.yaml` within each validation folder with the existing
`evaluate_native_nu2flow.py` for the **original nu2flow** count-uncertainty
ratio, original4×4joint bins, kreg7. Its Raw arm is this checkpoint's actual
unweighted policy; General/Mix arms are separate post-hoc diagnostics. Compare
Raw across the pre-DGPO and later caches, not post-hoc Mix against Raw.

The classifier `run.py ... evaluate` path separately provides split-sample
closure and data/response-MC covariance (300paired response bootstraps),
neutrino pT/eta/phi kreg5 and joint kreg7. Do not mix its weighted-shape metric
with the native count-uncertainty metric. Original native nu2flow has no
response-MC component; that omission is not zero uncertainty. Top-mass SVD
currently fails the declared reference-bin support check and must not be
advertised as validated without resolving that separately. Preserve the
original binning/selection limitations in NU2FLOW_ALIGNMENT.md.

## Manual launch

From `~/ml_pipeline`, after you prepare the Shifter environment and existing
16-GPU Ray cluster (four4-GPU nodes):

```bash
shifter python3 experiments/ttbar_dgpo_general/run.py config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml
```

Optional preparation only:

```bash
shifter python3 experiments/ttbar_dgpo_general/run.py config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml --check
```

Resume explicitly, retaining optimizer and reward history:

```bash
shifter python3 experiments/ttbar_dgpo_general/run.py config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml --resume /pscratch/sd/y/yiren/Preference_Optimization/ttbar-dgpo-general70-top-oracle30-full-mg5-run-01/checkpoints/last.ckpt
```

The launcher refuses an absent/smaller Ray cluster; it will not silently use a
single node. No16-GPU throughput or GPU-memory-fit measurement has been made
for this new arm. CPU checks cover the actual reward/update integration,
feature/chunk parity, cached mixture parity, periodic refit installation,
checkpoint pairing, and the native two-rank accumulation regression.

## Ray preprocessing transport

EveNet and ml_pipeline both contain `preprocessing.preprocess`. A Ray actor
deserializes dataset functions before entering `train_worker`, so changing
worker import order alone cannot fix a serialized reference to that ambiguous
module. `preprocessing_bridge.py` transports a uniquely named wrapper and loads
the exact pinned EveNet source by absolute path. The upstream process factory
still determines metadata, shapes and dropped columns, including EXTRA fields.
The classifier cache/refit reader uses the same explicit source loader.

`test_preprocessing_bridge.py --evenet-root PATH --ray` checks original-source
parity, a fresh process with ml_pipeline first on its import path, and a two-row
CPU Ray TorchTrainer dataset. It does not instantiate or update a physics model.
This verifies transport, not 16-GPU capacity or training throughput.
