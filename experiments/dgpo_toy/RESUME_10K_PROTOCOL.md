# Full-state3000->10000 continuation — declared before execution, 2026-09-21

User asks to continue from3000: slow progress does not establish a wrong method.
Restore EACH arm's step3000 model, AdamW moments/clock and training RNG from
`artifacts/dgpo_toy/fixed_reward_joint3_long_v1/*_state.pt`. Apply only7000
NEW updates, finishing at total10000. No replay, optimizer reset, reference
recentering, classifier fit, seed change or reward transformation.

Same fixed joint3 classifier/source, AdamW1e-4, K8/M4, batch64, DDIM20,
t uniform[0,.7], no additive KL/refit/EMA, monitor every25. DGPO's gate keeps
the ORIGINAL pretrained reference, not the resumed policy. Both DGPO and
pathwise continue, same training stream and update budget, not compute matched.

Validate resume source/config/clock/history/endpoint weights and reproduce the
last monitor reward exactly before any update. Unit test split/resume against
uninterrupted training for both arms: model, optimizer, RNG and complete history
must agree exactly. Keep old artifacts intact in separate output directories.

Primary unchanged: total10000-minus-initial fixed reward gain>=.10 and lower95%
context-cluster bound>0, on NEW endpoint stream93017 (4096 contexts x8).
Also report10000-minus3000 on this SAME new paired panel as a continuation
diagnostic. Monitoring retains stream90017; training RNG continues from3017.
Known joint statistic, reward std, gate, clipping, ESS and gradient metrics
remain secondary. No best-checkpoint selection; stop at10000 or nonfinite/error.
Save full state every1000 and at the end, without changing the optimizer path.

Same labels: DGPO pass=`reward_improves`; only pathwise passes=
`dgpo_transfer_deficit`; neither=`inconclusive_actionability`. Report actual
gains and intervals as well: an operational threshold is not the same as
evidence of a positive effect. A pass answers learnability at this budget,
not accurate ratios, classifier closure or matched low-ESS causality.

Runtime budget approximately10 local minutes including implementation checks;
no NERSC/W&B/production edits. Output fixed_reward_joint3_resume10k_v1.
No automatic budget extension or LR/architecture change after seeing results.
