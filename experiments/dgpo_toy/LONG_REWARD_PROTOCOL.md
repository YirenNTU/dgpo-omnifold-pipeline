# Fixed reward length extension — declared before execution, 2026-09-21

User requests training longer after the300-update plateau. Extend both existing
arms to3000 updates, changing ONLY the policy training horizon. Same fixed joint3
classifier, source generator, seed17, LR1e-4, AdamW, K8/M4, batch64, DDIM20,
monitor every25 steps, no additive KL/refit/tempering/EMA. No new classifier fit.

The old policy files lack optimizer states. Do NOT weights-only resume and call
it the same trajectory. Replay each arm from the original source, require exact
step300 weights AND full300-row history agreement with the saved short run,
then continue inside the SAME optimizer/RNG instance through step3000. Stop
the extension on mismatch. Save full model/AdamW/RNG/history states this time.
Do not overwrite either prior experiment's outputs.

Primary remains final mean fixed-reward improvement>=.10 logits with lower95%
context-cluster CI>0. Endpoint uses NEW seed92017,4096 contexts x8 candidates;
training3017 and monitoring90017 remain unchanged. The new endpoint stream does
not alter training. Compare300/1000/2000/3000 using the fixed monitoring panel
descriptively; no best-policy selection, no interim stopping for favorable data,
no unplanned extension past3000. Stop on nonfinite values or invalid replay.

Same decision labels: DGPO pass=`reward_improves`; only pathwise passes=
`dgpo_transfer_deficit`; neither passes=`inconclusive_actionability`. Retain
the approximate ratio-fidelity caveats. No claim of truth closure, low-ESS
causality, training-seed replication, or production generalization.

Question: was300 updates too short to observe meaningful fixed-reward gain?
If longer learning succeeds, the300-step plateau is not a permanent inability.
If neither succeeds, added budget up to3000 is insufficient at this setup;
it does not prove that arbitrary longer training cannot help.

Local-only expected runtime several minutes. Reports and checkpoints written
under `artifacts/dgpo_toy/fixed_reward_joint3_long_v1`. No production/W&B changes.
