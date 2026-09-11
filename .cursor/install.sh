#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the Ztautau EveNet / DGPO / OmniFold pipeline.
#
# The production workflow runs on NERSC Perlmutter (GPUs + SLURM + Ray + a
# shifter container image). This script instead prepares a lightweight,
# CPU-only Python environment that is sufficient to import the code and run the
# repository's validation test suite on a Cloud Agent VM.
#
# It deliberately does NOT rely on the Ubuntu apt mirror (which is not reachable
# from Cloud Agent VMs). Python itself is provided by `uv`, whose managed
# interpreters ship the development headers that native builds need.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[install] Repository root: $REPO_ROOT"

# 1. Ensure uv is available (manages Python + venv without apt).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[install] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[install] uv $(uv --version)"

# 2. Create the virtualenv with a managed CPython 3.12 (idempotent).
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "[install] Creating .venv with managed CPython 3.12..."
  uv venv --python 3.12 "$REPO_ROOT/.venv"
fi

# 3. Install pinned dependencies with CPU-only PyTorch wheels.
#    (Cloud Agent VMs have no GPU; the CPU wheels keep the install small.)
echo "[install] Installing Python dependencies (CPU PyTorch)..."
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r "$REPO_ROOT/evenet_dgpo/requirements.txt" \
  pytest

# 4. Best-effort native extension. `torch_linear_assignment` is only imported by
#    the supervised diffusion training entrypoint (evenet/train.py) and is
#    normally provided by the NERSC shifter image. Building it requires a C/C++
#    toolchain that may be unavailable here, so failure is non-fatal.
if ! python -c "import torch_linear_assignment" >/dev/null 2>&1; then
  echo "[install] Attempting optional torch-linear-assignment build..."
  uv pip install torch-linear-assignment >/dev/null 2>&1 \
    && echo "[install] torch-linear-assignment installed." \
    || echo "[install] torch-linear-assignment unavailable (optional; needs a C/C++ toolchain + headers). Skipping."
fi

# 5. Convenience: auto-activate the venv and export the paths the code expects
#    in future interactive shells. Guarded so it stays idempotent.
MARKER="# >>> ml_pipeline env (managed by .cursor/install.sh) >>>"
if ! grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
  echo "[install] Wiring venv activation + PYTHONPATH into ~/.bashrc"
  cat >> "$HOME/.bashrc" <<EOF
$MARKER
export PATH="\$HOME/.local/bin:\$PATH"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi
case ":\${PYTHONPATH:-}:" in
  *":$REPO_ROOT/evenet_dgpo:"*) : ;;
  *) export PYTHONPATH="$REPO_ROOT/evenet_dgpo:$REPO_ROOT:\${PYTHONPATH:-}" ;;
esac
export TORCH_NCCL_TIMEOUT=180
# <<< ml_pipeline env (managed by .cursor/install.sh) <<<
EOF
fi

echo "[install] Verifying core imports..."
PYTHONPATH="$REPO_ROOT/evenet_dgpo:$REPO_ROOT:${PYTHONPATH:-}" python - <<'PY'
import torch, numpy, ray, lightning
print(f"[install]   torch  {torch.__version__} (cuda={torch.cuda.is_available()})")
print(f"[install]   numpy  {numpy.__version__}")
print(f"[install]   ray    {ray.__version__}")
PY

echo "[install] Done."
