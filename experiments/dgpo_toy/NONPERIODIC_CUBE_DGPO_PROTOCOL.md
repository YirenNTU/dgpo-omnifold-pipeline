# Second ablation: Fourier diffusion with a fixed learned Fourier critic

2026-09-24 user authorizes local training and proceeding despite Plain classifier
not meeting the strict plateau gate. This changes the question, not the recorded
classifier result: no claim that Plain can never learn.

Source: nonperiodic_cube_classifier_extended_v1/reference.pt (trained uniform
cube reference, not full truth). Frozen reward: actual/fourier_classifier.pt,
selected step6900,not final16000 weights. TestBCE.652326/AUC.648105.
No classifier refit, no supervised pretraining repeated, no EMA.

Compare raw vs Fourier diffusion conditioning, from exactly identical initial
velocities and DDIM samples. Both contain identical-sized linear adapter;
raw inputs zero; Fourier c basis[1,2,4,8],sin/cos. Last adapter zero initialized.
Same native DGPO code, same data/seed17,AdamW1e-4,wd.001,clip1,K8,4 t samples
in[0,.7],DDIM50,velocity-MSE surrogate coefficient1.1000 updates per arm.
No pathwise reward gradients and no true-KL claim.

Endpointseed82017,4096 random conditions x32 samples; monitor81017; structure
monitor83017,1024 x32 every100updates. These differ from classifier fitting
and preceding exploratory evaluation seeds. No endpoint-selected checkpoints.

Primary: independent context-paired mean fixed-reward gain and Fourier-minus-raw
95% interval. Prespecified material gain.01; failure-and-rescue requires raw
upper95%<.01,Fourier lower95%>.01,and contrast lower95%>.01. If raw gains too,
report acceleration/advantage,not rescue. One seed cannot establish robustness.

Also measure32-bin conditional parity MAE, single/pair sign moment max and corner
fraction. These detect target-structure improvement and low-order damage;
reward gain alone does not prove truth alignment. Actual diffusion marginal
imperfections may contribute to the learned reward despite ideal cube controls.

Store progress, per-arm model/optimizer/RNG/history snapshots, independent
endpoint arrays, source/critic immutability assertions, and elapsed time.
No remote work or W&B upload. No automatic continuation beyond1000steps.

Preflight passed: loaded selected critic provenance; exact velocity and sample
matching; finite native DGPO loss and component-gradient diagnostics for both arms.
