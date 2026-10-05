#!/usr/bin/env bash
# Run on the LOCAL machine. Updates code only; does not submit or start jobs.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
remote_host=${1:-yiren@perlmutter.nersc.gov}
remote_dir=${2:-/global/homes/y/yiren/ml_pipeline}
cd "$repo_dir"
files=(
  config/train_diffusion_relation_context.yaml
  config/train_diffusion_relation_relations.yaml
  evenet_dgpo/evenet/network/body/relation_conditioning.py
  evenet_dgpo/evenet/network/body/visible_conditioning.py
  evenet_dgpo/evenet/engine.py
)
for item in "${files[@]}"; do
  test -f "$item" || { echo "Missing local source: $item" >&2; exit 1; }
done
# Preserve path hierarchy, never delete remote files or transfer artifacts/toys.
printf '%s\n' "${files[@]}" | rsync -av --itemize-changes \
  --exclude-from=NERSC/upload-excludes.txt --files-from=- \
  ./ "${remote_host}:${remote_dir}/"
