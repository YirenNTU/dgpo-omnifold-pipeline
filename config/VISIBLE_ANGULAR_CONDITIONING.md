# Visible angular Fourier conditioning

Implemented, not yet trained or evaluated on NERSC.

`dgpo_h4_angular_conditioning_1110.yaml` enables this branch. The shared
`train_diffusion_nersc.yaml` leaves it disabled so historical runs stay unchanged.
Add the following under an experiment's existing `network.Body.PET` to opt in:

```yaml
visible_angular_fourier:
  enabled: true
  theta_source: Part_eta
  phi_source: Part_phi
```

Set `enabled: false` for the baseline. Harmonics are fixed at **1, 2, 3, 4**.

For every observed token, use physical `Part_eta` to obtain
`theta = 2 atan(exp(-eta))` (implemented without exponential overflow), and use
physical `Part_phi` in radians. Concatenate sin/cos of k theta and k phi into
16 features. No z-score or inverse-CDF transform is applied to these angles.
Keep the original normalized input path unchanged. A zero-initialized,
bias-free 16-to-PET-hidden projection adds a residual immediately after the
PET feature embedding, before its existing attention blocks. The original
geometrical local-point inputs remain unchanged. This is an input embedding
ablation, not per-layer FiLM or a change to the diffusion time embedding.

## Information boundary

- Only raw observed `x` and its `x_mask` feed the new branch.
- Never read `x_invisible`, truth angles, generated candidates, labels, or a
  hand-designed target observable to build its features.
- Invisible-token residuals are zero; information reaches them through existing
  attention. Their ordinary noisy diffusion state remains an ordinary model input.
- Enable the path for neutrino denoising training and neutrino sampling. Do not
  enable it for visible-event reconstruction, where clean visible angles would
  reveal what is being generated.
- The ratio backbone uses the same observed-only branch when its checkpoint
  architecture enables it. This does not add candidate Fourier features.
- Padded tokens are cleared before trigonometry and after feature construction.
- The input dataset must actually contain observed/reconstructed quantities in
  these named fields. Code cannot detect truth values mislabeled as observed data.

## Checkpoints and training

The defaults below describe the original input-linear branch. The optional
fully unfrozen three-arm continuation and output-MLP adapter are documented in
[DIFFUSION_FOURIER_INTEGRATION.md](DIFFUSION_FOURIER_INTEGRATION.md); that experiment
gives the branch a separate optimizer without freezing PET.

The new projection belongs to `PET`, so it inherits PET's optimizer group and
freeze status. PET must be trainable for it to learn. Existing weights can be
loaded through the existing partial/weights-only loader; the missing projection
stays zero and therefore initially preserves predictions. Use fresh optimizer
state when adding this parameter to an old checkpoint. Resume a trained Fourier
checkpoint with the same enabled configuration; disabling it changes the model.
No EMA selection, learning rate, loss, dataset, or sampling setting is changed
by this feature. No jobs have been submitted. Multi-GPU execution has not been
validated here.

Local tests:

```bash
PYTHONPATH=evenet_dgpo python -m unittest evenet.network.body.test_angular_conditioning -v
```

The tests cover formulas, all four harmonics, angle periodicity, padding,
extreme eta, zero initialization, gradients, PET checkpoint compatibility and
source-routing contracts. Empirical improvement remains to be tested with a
matched no-Fourier control; toy results do not establish the real-data bottleneck.
