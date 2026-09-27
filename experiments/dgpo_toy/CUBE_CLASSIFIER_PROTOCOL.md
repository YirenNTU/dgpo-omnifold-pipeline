# Learned reward transfer on the hard conditional cube

Prepared 2026-09-22; not launched. User submits training jobs.

Question: Does replacing the successful fixed categorical oracle with a learned
Fourier classifier prevent native DGPO from absorbing reward?

Source: cube_difficulty_sharp4_v1/pretrain/best_pretrain.pt. Reuse completed
300-step velocity_mse_1 oracle checkpoint as the matched control. No diffusion
pretraining, changed sampler, reward transformation, or extra policy arm.
Keep source, policy seed17, fresh AdamW, LR, K8, M4, DDIM50 and coefficient1.

Classifier: raw 3 coordinates, continuous condition, coordinate-wise sin/cos
at k*pi/2 for k1..4; 128-wide three-layer GELU MLP. No parity/target feature.
Balanced truth versus actual source-generator BCE; independent train/val/test
panels seeds730017/740017/750017. Counts inherit source config. 5000 fit updates,
LR3e-4, best validation BCE selection. Existing function-preserving hidden
standardization at step50. This is a toy Fourier classifier, not production H4.
Validation trace and selected step expose undertraining; a best at the budget
boundary is flagged. No saturation claim based solely on budget. Test unused
for selection. Freeze classifier during all policy updates.

Primary: paired independent learned reward gain at300, seed690017, context-level
95% CI. A positive lower bound supports reward absorption, not truth closure.
Secondary: oracle reward, preferred mass, full8mode TV in8condition bins,
near-corner fraction, centroids/widths, global ratio ESS, strict candidate-pair
agreement excluding oracle ties, within-context centered cosine and eligible
context fraction. Compare both final policies with the SAME frozen classifier.
Report paired learned-minus-oracle contrasts. Save evaluation arrays/checkpoints.

If learned reward improves without oracle/mode improvement: reward can transfer,
but its target differs or is exploited. If neither improves while classifier fit
is unresolved: no bottleneck conclusion. If well-fitted learned reward fails
while oracle succeeds: supports a reward/interface limitation in this toy, not
proof that ESS causes real-case failure. If both work: toy lacks the real failure.

Important confound: learned density ratio targets actual neural generator;
oracle uses configured categorical mixture. This intervention changes reward
scale, smoothness and target approximation, not solely classifier error or ESS.
Single training seed; paired intervals quantify evaluation sampling only.
Offline W&B by default. Monitor every50; no automatic cloud upload.
