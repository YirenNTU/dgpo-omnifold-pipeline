# H4 functional trust-region boundary test (`h4trust01`)

## Question

Can the fixed `h4rep01` seed-2 H4 reward reduce a cold best-response H4 gap
before the policy leaves its calibrated local region when the paired reference
is enforced as a constraint instead of an additive gradient?

This tests the leading mechanism left by `h4rep01`: its late coefficient-1
reference removed 85–96% of the H4 projection, while historical refits showed
that merely resetting that reference gradient was not sufficient for closure.

## Frozen state

- Full-state resume from the `h4rep01` seed-2 step-0 snapshot.
- The same installed four-member H4 reward, paired reference policy, seed
  bundle, ESS15 tempering, `K=8` calibrated leave-one-out advantage, policy LR,
  native AdamW state, and fixed cold-audit panel.
- No classifier refit, optimizer reset, reward transformation, or secondary
  projection constraint.
- Redundant train-distribution and diffusion validation panels are disabled;
  the independent cold H4 audit is the declared endpoint.

## Treatment

- Optimize the unchanged H4 DGPO candidate loss with additive reference
  coefficient `0`.
- Constrain cumulative policy displacement from the step-0 paired reference to
  `velocity_mse_ratio <= 1e-4` on a fixed shared-noise 4096-event probe.
- Test the proposed AdamW displacement at absolute scales
  `1, 1/2, ..., 1/64`.
- If no positive scale is feasible, restore policy parameters, all AdamW
  moments and step counters, the scheduler, and EMA. Run the endpoint audit and
  stop. The run has at most ten proposals.

The proposal clock and accepted-update clock are logged separately. A rejected
proposal cannot silently become an optimizer update.

## Decision

The fixed seed-2 baseline is `gap0 = 0.3711905`. The primary endpoint is the
cold saturated H4 gap at the first rejected boundary, or after proposal 10 if
all proposals are accepted. Every audit must train for at least 1000 optimizer
updates.

```text
pass iff endpoint_gap - 0.3711905 < 0
```

The matched `h4rep01` coefficient-1 control worsened to `0.4083256` at step 10.
The endpoint also runs the installed-versus-fresh gradient diagnostic. Positive
alignment with an improved cold gap supports the hard-constraint formulation;
loss of alignment points back to classifier rotation inside the tested radius.

## Run

```bash
shifter python3 scripts/train_dgpo_h4_function_trust.py
```

The launcher requires the existing 16-GPU Ray cluster and online W&B. Training,
trust acceptance, endpoint audit progress, and the final decision are written
to W&B run `h4trust01` in real time.
