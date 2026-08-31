"""Shared posterior diagnostics vendored from EveNet-private.

``core.py`` and ``tarp.py`` are kept implementation-identical to
EveNet-private commit ``61d0a771c719d161a3634fde57fd4947a1d3a64b``.  Keeping
the statistical implementation local makes NERSC ``ml_pipeline`` deployments
self-contained while preserving metric comparability with the reference DGPO
run.
"""

from .core import Metric, Prediction
from .tarp import (
    BinnedTarpResult,
    GRID,
    TarpResult,
    evaluate_tarp,
    evaluate_tarp_binned,
    holm_adjusted,
)

__all__ = [
    "BinnedTarpResult",
    "GRID",
    "Metric",
    "Prediction",
    "TarpResult",
    "evaluate_tarp",
    "evaluate_tarp_binned",
    "holm_adjusted",
]
