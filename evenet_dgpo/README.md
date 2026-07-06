# EveNet DGPO Compatibility Layer

This directory vendors the `DGPO_review` training code needed to fine-tune the
EveNet generation head inside `ml_pipeline` without changing the current
prediction and export I/O.

The intended workflow is:

1. train the classifier with the current EveNet pipeline
2. export classifier-driven `event_token` / `object_token` inputs for the AD stack
3. reuse the fine-tuned diffusion checkpoint as the DGPO starting point
4. launch either backend through `scripts/train_neutrino_backend.py`

Ztautau-specific logic should live under:

- `evenet_dgpo/RL/DGPO_neutrino/domains/`

That keeps the DGPO core mostly generic while isolating:

- feature-space assumptions such as `theta/phi`
- back-to-back topology helpers
- future Ztautau-only reward / constraint / diagnostic glue

Launcher example:

```bash
python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config /path/to/train_pretrain.yaml
```
