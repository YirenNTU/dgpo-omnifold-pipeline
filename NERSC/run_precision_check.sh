#!/usr/bin/env bash
# User launches once inside an existing 16-GPU Ray allocation. No submission.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
exec shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_precision.py \
  --checkpoint /pscratch/sd/y/yiren/Ztautau/film_strength_real_01/raw_policy.pt \
  --panel /pscratch/sd/y/yiren/Ztautau/film_strength_real_01/panel.pt \
  --output /pscratch/sd/y/yiren/Ztautau/precision_screen_epoch286_01 \
  "$@"
