#!/bin/bash
# Compatibility entry point. Source this file so RAY_ADDRESS and the selected
# head-node variables remain available in the current shell.

_ML_PIPELINE_NERSC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=start_interactive_ray.sh
source "${_ML_PIPELINE_NERSC_DIR}/start_interactive_ray.sh"
unset _ML_PIPELINE_NERSC_DIR
