# Full-state3000->10000 continuation — completed, 2026-09-21

## Outcome: delayed learning, not inability to improve fixed reward

**`reward_improves`**: DGPO passes the unchanged primary endpoint at10000.
The identical low-ESS classifier does supply learnable reward to this neural
diffusion. An early300/3000-update plateau is not a permanent failure, and this
toy is no longer evidence that low ESS necessarily prevents DGPO improvement.
It does not prove low ESS is irrelevant to efficiency or explain production.

Both arms restored their exact3000 model, AdamW moments/clock and training RNG,
then applied7000 NEW updates. Their saved monitor boundary reproduced exactly.
The original pretrained reference remains inside the DGPO gate; no recentering,
additive KL, critic fit, LR change, seed change or reward transformation occurred.
Frozen classifier/source weights and buffers stayed unchanged. Resume equivalence
to uninterrupted training is tested for BOTH estimators across model, optimizer,
RNG and full history. No prefix was replayed in this round.

Protocol: [RESUME_10K_PROTOCOL.md](RESUME_10K_PROTOCOL.md).

## Primary endpoint — NEW4096 contexts x8, stream93017

Initial mean reward=-2.159617, std=.834286. Confidence intervals use paired
context-cluster means, not independent candidates. They quantify evaluation
uncertainty for this one training seed, not seed-to-seed reproducibility.

| Arm | Final mean reward | Gain vs initial |95% CI | Gain vs step3000 |
|---|---:|---:|---|---:|
| DGPO | -.719222 | +1.440396 | [1.394523,1.486268] | +1.438346 |
| Pathwise | +.371307 | +2.530924 | [2.445954,2.615894] | +2.470616 |

Continuation-gain intervals: DGPO[1.392568,1.484123]; pathwise
[2.385509,2.555723]. Both total gains exceed the original+.10 threshold with
positive lower confidence bounds. No best intermediate policy was selected.

Known joint-statistic gains vs initial: DGPO+.452082,
CI[.444651,.459514]; pathwise+.611533,CI[.602728,.620338]. This is a toy
third-order statistic, not physics performance, fresh-classifier closure or
proof of matching all truth structure. Pure reward optimization need not recover
the truth distribution or preserve its lower-order marginals.

## Timing of the improvement — same monitoring panel throughout

| Total updates | DGPO reward gain | Pathwise reward gain |
|---:|---:|---:|
|3000|-.000371|+.076879|
|4000|+.031433|+2.487037|
|5000|+.366198|+2.487107|
|6000|+.714634|+2.487312|
|7000|+1.329634|+2.487363|
|8000|+1.424557|+2.487342|
|9000|+1.213159|+2.487343|
|10000|+1.458546|+2.487332|

First monitored crossing of gain>=.10 with positive lowerCI is step4325 for
DGPO and3050 for pathwise. These are descriptive, repeatedly inspected crossing
times, NOT additional confirmatory endpoints or stopping rules. Do not map toy
update counts to production epochs/steps: sampling budgets and models differ.

![Recorded trajectories](../../artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1/trajectory.png)

## A new limitation becomes visible after learning starts

DGPO is not monotonic: the monitor reaches gain3.169736 at7425, drops sharply
to .066452 at7525, then partially recovers to1.458546 at10000. The peak is a
descriptive observation, not a selected outcome.
For steps3001–10000, DGPO clips2487/7000 gradients; for the final1000 updates,
999/1000 clip. Late preclip gradient norm median9.6788 (maximum94.6489);
late gate-saturated fraction median .246094, maximum .410156. Earlier through
3000, both clipping and gate saturation were absent. These are new late-phase
phenomena, not explanations of the initial plateau, and association does not
establish that clipping/gate saturation caused the later reward decline.

Pathwise clips253/7000 new updates,0/1000 late. Its late gradient norm median
is approximately1.3e-21 while monitored reward is nearly constant. That records
a plateau/vanishing gradient in this parameterization; it does not establish
a global optimum or distinguish feature saturation from other causes.

Final frozen-weight ESS/N is .058775 for DGPO and .123009 for pathwise,
versus .000486 on this new initial panel. These are weights from the SAME
old classifier, not newly fitted p/q_current ratios. The larger original
classifier confirmation panel's ESS/N=.001590 remains the setup evidence;
rare tails make smaller-panel ESS fluctuate.

## Decision and scope

- Supports the user's hypothesis that an early plateau can reflect delayed
  learning/low efficiency rather than an inability to improve fixed reward.
- Rules out treating this toy's initial low ESS as a sufficient permanent
  barrier to DGPO reward learning. No matched high-ESS intervention ran, so
  low ESS as a causal slowdown remains unresolved.
- Does not prove the DGPO mathematical objective is correct for truth closure,
  production training needs10000 steps, or all higher-order discrepancies close.
- The next limiting factor is preserving/stabilizing the learned gain after
  onset, given the late DGPO recoil and changed gradient/gate regime. Keep that
  distinct from the now-falsified claim of permanent early stagnation.

The earlier300/3000 reports retain their original observations; their permanent
failure interpretation is superseded by this continuation. A correct next round
should use the saved full states and target this specific late-phase question,
not silently modify reward, optimizer and horizon together.

## Artifacts and verification

Directory: `artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1`.
`report.json` includes both complete10000-row histories, resume verification,
new-panel initial and3000 baselines, final and incremental endpoints.
`progress.jsonl` contains NEW monitored steps only. `*_state.pt` retains final
model, AdamW step10000, RNG and10000-row history; `dgpo.pt`/`pathwise.pt` match
the full-state endpoint weights exactly. Earlier source directories unchanged.

50 tests pass; compile and whitespace checks pass. Full-state files verified
read-only after completion. Runtime441.78s (DGPO200.13s,pathwise240.86s plus
baseline). No production changes, W&B writes, NERSC jobs or classifier fits.

Elon-workflow outcome: one training-budget variable changed; about442 local
CPU wall seconds resolved the fixed-reward learnability question in this toy
and exposed a distinct late-stability question. Better next round: preserve
this separation and the full continuation state rather than call every flat
early window a permanent failure.
