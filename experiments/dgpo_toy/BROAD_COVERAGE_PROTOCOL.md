# Broader joint-target pretraining — 2026-09-22

One new diffusion, original architecture/AdamW/data counts/seed17/early-stop20,
raw weights/noEMA. Training distribution changes kappa8->2, keeps structured
mass.9, correct phase centers, and same ideal univariate/bivariate marginals.
No teacher or phase features in the denoiser. Dataset and output separate from truth.
Comparator: existing truth_diffusion_earlystop_v1 selected raw checkpoint.
Same32,768/8,192/16,384 split counts; ordinary sample velocity loss. No RL/classifier.

Independent evaluation4096 conditions,K128,seed370017 shared Gaussian noises.
Nested K8 uses first8. Region: ALL4 wrapped triple phases within1radian of their
known true centers. This is one operational joint-region coverage metric, not
global support or complete conditional-distribution coverage. Primary paired anyK8
probability gain lower95CI>0. Low-order mean<.05,varerror<.15,paircov<.05.
Report candidate-hit frequency and anyK128; saturated anyK128 alone is not proof.
Known true/broad samplers independently validate target regions, never substitute
for learned model measurements. Endpoint56moments compared with ORIGINAL kappa8
truth, not the broadened training target. No automatic RL launch even if gates pass.
Single seed; postulated benefit can fail if diffusion does not learn broadened joints.

Run: python -u -m experiments.dgpo_toy.broad_coverage --output artifacts/dgpo_toy/broad_joint_coverage_v1
