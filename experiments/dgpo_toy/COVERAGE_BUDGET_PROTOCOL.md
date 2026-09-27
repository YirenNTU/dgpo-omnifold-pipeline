# Finite-rollout coverage pilot

Question: does greater within-condition candidate coverage make the frozen strong
reward more actionable, beyond averaging more independent K8 loss calls?

Source: `artifacts/dgpo_toy/direct_transport_joint_v1/reward.pt`, **initial**
base diffusion weights, not the completed10k policy. Frozen joint3 classifier and
fixed physical transport a=.6. No new truth training, classifier fits, oracle
training reward, pathwise reward or external proposal. Native base-coordinate
velocity MSE coefficient1 is a surrogate, not exact endpoint KL beta1.

Arms, each100 optimizer updates, same64 contexts/update:

| Arm | Candidates | Loss calls/update | Candidates/update |
|---|---|---|---|
| A | K8 | 1 | 512 |
| B | K128 | 1 | 8192 |
| C | 16 independent K8 groups | 16, averaged | 8192 |

All arms have fresh identical AdamW state, source LR/WD, clipnorm1, DDIM20,
M4 diffusion times. Each C call computes its complete original nonlinear gate
before loss averaging. Reference penalty is averaged, not multiplied by16.
Only one clipping and AdamW step per optimizer update in every arm.

Common conditions and128 initial candidate noises couple B/C; A gets the first8.
Once policies diverge, generated values naturally differ. B uses native shared
t/eps within its128 group; C has independent t/eps per8 group (first call matches
B). Candidate denoising-loss evaluations match B/C, but runtime/overhead and
number of distinct t/eps draws need not. Record both compute counters and time.
C groups are independent conditional on the shared contexts, not independent
context minibatches. K changes finite-K LOO/gate behavior: this is NOT a pure
coverage-only causal intervention and cannot isolate ESS alone.

All three are interleaved, evaluated at1/25/50/75/100 on the same fixed4096-context
K8 monitor. Final independent4096-context K8 panel supplies paired reward and
56moment bootstrap comparisons. Log rollout group-hit fraction, union-pool-hit
fraction, good-candidate counts, call-level ESS/gate means, actual aggregate
main/reference gradients, cosine/projection, clipping and displacement. Coverage
is the existing toy joint-phase-region hit, NOT distance to a specific truth draw.

Primary: independent fixed-reward gain at100 and paired B-A/B-C95% intervals.
Support for grouping/coverage requires both reward intervals positive and
greater measured loss-group coverage. Distribution-alignment evidence separately
requires decreasing56moment error without low-order degradation (meanabsmax<.1,
varianceerror<.15,paircov<.08). Do not select the best intermediate step.
If B and C improve similarly, more sampling rather than grouping is implicated.
If B gains coverage but not useful updates, coverage is not sufficient in this
regime;100-step failure is not proof of asymptotic impossibility. A single seed
is a pilot; paired evaluation intervals do not quantify training-seed variability.

Local JSONL and W&B offline telemetry are written in real time. No automatic
upload, production changes or extension to10k. Final weights and optimizer states
are retained; saved initial gradient vectors allow inspection. A new output
directory prevents mixing results. Full-state continuation uses --resume; per-step
training RNG is a deterministic function of the original seed and absolute step.
All three source steps/configs must agree; model and AdamW state are restored.

## User-authorized long continuation

Extend the completed100-update pilot to total10000, preserve all training choices.
Monitor at101 then every500 updates; reference remains original step0, never reset
to100. Fresh endpoint seed540017 avoids reusing the inspected100-step endpoint.
The long horizon is exploratory, chosen after seeing the pilot; it is not an
independent training-seed replication. Compare endpoints to original initialization
and between arms, not just against the best intermediate checkpoint.

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.coverage_budget \
  --resume artifacts/dgpo_toy/coverage_budget_v1 \
  --output artifacts/dgpo_toy/coverage_budget_10k_v1 \
  --steps 10000 --eval-every 500 --endpoint-seed 540017
```

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.coverage_budget \
  --output artifacts/dgpo_toy/coverage_budget_v1 --steps 100
```
