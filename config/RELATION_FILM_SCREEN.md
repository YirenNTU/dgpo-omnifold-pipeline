# Real-case relation-aware FiLM screen

Status: implemented/prepared, no jobs submitted or remote training started. This is a prioritized architecture hypothesis, not a proven highest-leverage solution.

## Question and matched arms

Do explicit observed pair relations improve joint generation beyond a same-size context adapter?

- `context`: input to the new adapter is existing 64-d event context concatenated with its first eight channels.
- `relations`: same context, but the extra eight values are mean/max over unordered valid visible pairs of cos(delta phi), opening-angle cosine, squared energy asymmetry, and squared pT asymmetry.

No truth, invisible candidate, tau assignment or diffusion time is read by this branch. Symmetric pair features and pooling preserve visible-slot permutation invariance; phi periodicity is explicit. Padding, zero/one valid particle and empty events are handled. The design is not claimed fully Lorentz invariant or detector-rotation invariant. Raw energy/pT are clamped nonnegative; valid nonfinite inputs are rejected. Features summarize all visible pairs, not identified tau decay systems. This compact summary may discard useful relation information; a null result does not rule out all relational architectures.

Both arms use the same width64 two-layer SiLU encoder plus LN and three zero-output projection heads. They add per-block attention/MLP scale and shift to existing global FiLM. Same parameter shapes, values and nominal/active parameter count at construction; pairing arithmetic costs extra FLOPs, so this is equal training steps, not equal wall-clock. Additional branch construction preserves global RNG. Source global FiLM is never reset. No new multiplicative gate creates a double-zero gradient trap.

## Source and training

Use the pinned raw epoch286/global_step22386 snapshot from diagnostic 23522203:
`/pscratch/sd/y/yiren/Ztautau/film_strength_real_01/raw_policy.pt`.
This is an explicit common source, not a claim that epoch286 is the best checkpoint. Strict migration allows only new relation-adapter weights to be absent. All other keys/shapes/dtypes must match and source values must be finite. Already-trained adapter sources are rejected. EMA is disabled; weights-only fresh optimizer/scheduler. Existing scale remains trainable in BOTH arms; do not combine this experiment with gain reduction/removal.

Inherit the existing matched 50-epoch supervised recipe: 16 GPUs, batch2048/GPU, paired diffusion seed420024, model seed42, FP32, all backbone weights trainable, body LR2e-5 and head/visible-conditioning LR1e-4, five-epoch warmup and cosine50, no early stop within budget. No DGPO objective change or classifier training. Adapter lives inside the existing visible_conditioning optimizer component. Separate output paths and W&B IDs: relcontext01 / relrelations01.

## Generation endpoint

Enable the existing JointCoverage callback at training start and completed epoch50. Same saved 1024-event panel, K32, DDIM20 legacy sampler, CPU noise seed42017, 16 GPU shards. Save raw generated samples, metrics and plots using the existing callback.

Primary: epoch50 relation-arm joint-radius W1 minus context-arm joint-radius W1 (negative favors relations). Joint radius is max(acoplanarity, acollinearity). Read `result.topology.joint_radius.w1_radians` from each `checkpoints/joint_coverage/epoch-0050.json`; compare start panels first. The callback's paired CIs compare each arm against its own start, NOT the two arms against one another. A direct arm-vs-arm event-bootstrap comparison is needed before a significance claim.

Secondary: marginal W1/CDF/tails, joint coverage, invalid direction counts, final-five validation velocity MSE, existing total scale/shift RMS and new relation_scale_rms/relation_shift_rms. An improvement in MSE alone is insufficient. One seed and reused exploratory panel cannot establish robust generalization; successful architecture requires a separate confirmation panel and subsequent fresh adequate H4 audit for any DGPO-closure claim. More signal amplitude is not itself a success criterion.

## User launch, sequential arms on existing allocation

Sync changes into the existing remote ml_pipeline. Root rsync must use `--exclude-from=NERSC/upload-excludes.txt`; do not upload toy code/artifacts or saved classifier checkpoints.

Run each arm separately using the same existing 16-GPU allocation (do not launch both simultaneously on 16 GPUs):

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_relation_context.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_relation_context_01/ray_results
```

Then:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_relation_relations.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_relation_relations_01/ray_results
```

Local tests cover nonzero-source step0 equality, RNG preservation, equal parameter counts and shared initialization, trainable output gradients, strict checkpoint migration, paired geometry/permutation/periodicity/padding, and resolved 16-GPU matched configs. Full Lightning optimizer integration and NERSC execution remain unverified in the local environment lacking Lightning.


### Missing remote overlay recovery

If launch raises FileNotFoundError for train_diffusion_relation_relations.yaml, run locally:

```bash
bash NERSC/sync_relation_film.sh
```

The helper uploads exactly the two new configs, the relation module and its visible-conditioning/engine integration into the existing remote repository. It uses upload-excludes, performs no deletes and submits no jobs. It assumes the pre-existing global-FiLM base/overlay code is already synced. Pass a different SSH host and remote directory as its first and second arguments if needed. After successful transfer, rerun the original launch from the remote repository.

### Supervised source loader correction

The initial relation loader rejected any extra state entry, unlike the already successful raw diagnostic loader. Supervised sources can contain auxiliary `famo.w.*` loss-balancing state installed after model loading. The corrected loader ignores only those entries when absent from the target, while rejecting every other unexpected key, missing shared tensor, shape/dtype mismatch and nonfinite shared tensor. Errors now list missing/unexpected keys. The pasted relrelations01 failure happened at configure_model, before any training iteration; its generic old error did not list keys, so FAMO is the likely cause based on this verified loader inconsistency, not a directly inspected remote checkpoint. After syncing the fix, restart the original command from the common source; do not restore the failed Ray trial. Local regression checks: 24 passed including FAMO acceptance and rejection of other extra keys. Actual remote reload remains to be verified.

### Confirmed recovery source and 100 additional epochs

User inventory confirmed only full-state relation checkpoints at epoch6,12,13; no epoch49 checkpoint. The resume overlay now explicitly requires epoch13/global_step1092's latest saved epoch, runs to completed114 (100 new epochs), and logs the recovery origin. It preserves optimizer/scheduler counters but extends cosine horizon50 to114. Coverage retains the original paired baseline and adds an endpoint evaluation at114, alongside scheduled50/100. Its callback permits changing only `final_completed_epoch` on restore; panel, noise and sampling configuration remain strict. Both original relation arms and the resume save real last.ckpt files going forward. This cannot reconstruct the lost epoch49 weights. Sync both resume YAML and joint_coverage.py before retrying the same launch. No job submitted by the assistant.
