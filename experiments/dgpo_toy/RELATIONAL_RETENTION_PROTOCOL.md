# Minimal conditional reward-retention failure: local iterative campaign

Authorized by the user on 2026-09-29: start local toy iterations until a minimal
failure is identified. No remote jobs, uploads, or production edits. One seed.
Up to20 reviewed iterations; stop on a falsifiable reproduced-and-rescued toy
mechanism or report unresolved. Do not claim real-case attribution from a toy.

## Round1, declared before execution

Reuse completed relational classifier study's selected relative critic. Truth
law, critic, source, samples, native loss, coefficient1 half velocity MSE,
AdamW1e-4, and batches remain matched. Raw-visible versus relative-feature
three-block FiLM corrections start with bitwise-identical source generation.
The source has been pretrained on the uniform cube; the remaining conditional
parity/joint law is not learned there. Only frozen source decodes scalar c.

Run1000 policy updates per arm. Evaluate4096 independent IID contexts x8
candidates, ALL-sample mean, at0/1/5/10/20/50/100/200/300/500/750/1000. Maintain
independent evaluation RNG. Dense monitors do not advance training RNG.

Primary failure windows are5->20,20->100,100->300,300->1000, chosen before
seeing curves. Use simultaneous condition-bootstrap intervals over BOTH arms
and all windows. Require early gain lower95>.01, later change upper95<-.01,
and late gain minus HALF early gain upper95<0. This distinguishes actual gain
loss from plateau or slower growth, and avoids choosing a noisy peak.

If failure appears, seek a matched rescue and remove unnecessary components;
confirm chosen windows on a fresh evaluation panel, not a new training seed.
If no failure appears, report that fact before choosing a different single
mechanism. Do not increase truth complexity arbitrarily. Existing successful
toy and production aggregate results are not proof of condition interference.

Independent endpoint classifiers: same width128 relative-feature architecture,
32768/8192/16384 contexts,32k minimum/64k cap, exact minimum validation BCE.
Audits are explicit operations after reviewing fixed-reward results; fit caps
without plateau are inconclusive. They never alter policy, reward or reference.
Fixed reward improvement alone does not establish joint/closure improvement.

Telemetry is live local-offline W&B plus JSONL, no online upload. Checkpoints
include model, optimizer, RNG and history. Record every round decision before
launching the next. Runtime is bounded initially by a one-hour review point;
the scientific stopping criterion is evidence, not a forced positive result.

## User-added periodic fresh-refit branch

Implemented in `relational_refit.py`. The first comparison starts from round02's
same raw endpoint and residual classifier, with fresh cold fits at100/200 and
an endpoint at300. Every reward refresh uses negatives from the current policy
and recenters its velocity reference there; preserve actor AdamW/RNG. The
archived fixed control is accepted only after bitwise first100 replay of
weights, optimizer and RNG. Cross-refit reward means are not concatenated as
evidence: an unchanged judge and independent cold endpoint audits are required.

Training panels for refits use the original matched panel seed. Final audits
use a separately declared seed and identical32768/8192/16384 train/val/test
sizes, architecture, initialization, minibatches and validation-BCE selection
across arms. Paired fresh BCE/AUC contrasts are the primary endpoint. This is
the toy analogue of reward/reference refreshing, not full OmniFold unfolding.

After round05's review, round06 isolates reference recentering only. This is a
diagnostic control for fixed-reward optimization, not a claim of an exact
truth-targeting density-ratio objective. Compare with BOTH the archived fixed
control and round05's fresh-refit arm. The original retention-failure threshold
remains unchanged, even if an independent distributional improvement is found.
