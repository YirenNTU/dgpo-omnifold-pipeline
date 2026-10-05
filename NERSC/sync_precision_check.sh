#!/usr/bin/env bash
# Local upload of diagnostic files only; no job submission or deletion.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
remote_host=${1:-yiren@perlmutter.nersc.gov}
remote_dir=${2:-/global/homes/y/yiren/ml_pipeline}
cd "$repo_dir"
files=(
  scripts/diagnose_precision.py
  NERSC/run_precision_check.sh
  config/PRECISION_FAST_CHECK.md
)
for item in "${files[@]}"; do
  test -f "$item" || { echo "Missing source: $item" >&2; exit 1; }
done
printf '%s\n' "${files[@]}" | rsync -av --itemize-changes \
  --exclude-from=NERSC/upload-excludes.txt --files-from=- \
  ./ "${remote_host}:${remote_dir}/"
