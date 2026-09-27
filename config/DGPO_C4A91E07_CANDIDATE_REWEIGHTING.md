# c4a91e07 fixed-candidate counterfactual

The cross-run evidence review supporting this diagnostic is in
`artifacts/c4a91e07_review/CONSOLIDATED_EXPERIMENT_REVIEW_20260915.md`.
It shows that ESS control, ensemble size, refit cadence, AdamW state, and
classifier undertraining do not by themselves explain the observed
reward/physics disagreement.

This is a one-question experiment:

> On the fixed c4a91e07 step-1110 policy and the exact same K=8 candidates,
> does following the classifier's within-event ordering move the distribution
> toward truth before DGPO is involved?

The diagnostic loads the completed `c4a91e07_h4_reward_interface_v5` artifacts.
It fits no classifier, generates no candidate, and performs no policy or
optimizer update. For every event it evaluates the self-normalized finite-pool
counterfactual

```text
w(candidate | event) proportional to exp(alpha * classifier_logit).
```

For raw H4 logits the predeclared production endpoint is `alpha=0.75`, matching
the H4 reward-interface arm. The restored legacy stack score already contains
all saved iteration coefficients and tempering, so its matched endpoint is
`scale=1.0`; it is not tempered a second time. The other points test whether
the result changes consistently with signal strength. A hard winner arm
measures the ranking ceiling and is not a proposed training rule.

The H4 score is the mean logit of the existing two-seed, two-fold ensemble. The
four member scores are also evaluated separately. The old non-Fourier reward
stack is restored directly from the same `c4a91e07/last.ckpt` and scored on the
same events and candidates. It is a matched historical control, not a newly fit
classifier.

The YAML fail-closes unless the primary contract remains the H4 ensemble at
`alpha=0.75`, with an independent-judge improvement, a lower mean target JSD,
and at least three of four improving target components required for the
`candidate_direction_useful` finding. W&B records this primary arm explicitly,
including its mean and fifth-percentile within-event ESS and mean winner mass;
later control arms cannot overwrite the primary summary.

The legacy arm is a historical direction control. Its original training split
was not constructed from the v5 five-way partition, so it must not be used to
claim an unbiased H4-versus-legacy generalization comparison.

Every arm reports:

- independent H4 judge AUC gap and balanced accuracy;
- all target, reconstructed-angle, and topology JSDs;
- the error of the target correlation matrix and component scales, using
  `theta`, `sin(phi)`, and `cos(phi)` features for periodic angles;
- truth-normalized four-delta-coordinate response proxies and event-paired
  bootstrap intervals;
- within-event ESS, entropy, and winner mass.

The response matrices in this fast diagnostic are only coordinate-level
proxies over the four generated tau deltas. They pool channels and do not
reproduce the downstream ROOT/RooUnfold observable definitions, selection,
efficiency, miss/fake, or overflow handling. The production response matrix
must be evaluated separately and is not used to decide the primary finding.

The central interpretation uses only the H4 ensemble at `alpha=0.75`:

- independent judge plus at least three of four target JSDs improve: the reward
  has a useful finite-candidate direction, so the next suspect is the
  advantage/gradient/policy transfer;
- the judge improves but target JSD does not: the H4 classifier objective is a
  physics-proxy mismatch;
- the independent judge does not improve: the candidate ordering is not
  reproducible even within the H4 family;
- mixed target changes: K=8 or event-region resolution is insufficient for a
  clean direction claim.

Run the read-only preflight and experiment from the active 16-GPU NERSC Ray
allocation:

```bash
shifter python3 scripts/diagnose_candidate_reweighting.py \
  config/dgpo_10pct_c4a91e07_candidate_reweighting.yaml --check-only

shifter python3 scripts/diagnose_candidate_reweighting.py \
  config/dgpo_10pct_c4a91e07_candidate_reweighting.yaml
```

The run logs live to W&B id `h4cfw001` and writes `report.json` plus the exact
candidate weights to
`/pscratch/sd/y/yiren/Ztautau/c4a91e07_candidate_reweighting_v1`.

This finite K=8 test cannot establish that the diffusion policy can represent
the reweighted mixture. If H4 passes here while a small signed policy step
fails, the evidence localizes the failure to classifier-to-DGPO transfer. If it
fails here, changing AdamW, refit cadence, or a trust region cannot repair the
underlying candidate direction.
