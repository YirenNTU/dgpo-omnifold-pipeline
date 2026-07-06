"""
Ray uses AF_UNIX sockets under ``RAY_TMPDIR`` (e.g. ``.../sockets/plasma_store``).

On Linux the full path must stay under ~107 bytes. Paths under NERSC ``$SCRATCH``
(or similar) are often too long and raise ``OSError: AF_UNIX path length...``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Ray appends something like: /ray/session_<ts>_<pid>/sockets/plasma_store (~70+ chars).
_MAX_RAY_TMPDIR_PREFIX_LEN = 36


def ensure_short_ray_tmpdir() -> None:
    """
    If ``RAY_TMPDIR`` is set and its length is unsafe, replace it with
    ``/tmp/r<uid>_<SLURM_JOB_ID|local>`` and update the process environment.

    Idempotent for already-short values. Does nothing when ``RAY_TMPDIR`` is unset
    (Ray's default is usually short enough).
    """
    current = os.environ.get("RAY_TMPDIR", "") or ""
    if not current:
        return
    if len(current) <= _MAX_RAY_TMPDIR_PREFIX_LEN:
        return

    uid = os.getuid()
    job = os.environ.get("SLURM_JOB_ID", "local")
    short = f"/tmp/r{uid}_{job}"
    logger.warning(
        "RAY_TMPDIR is %d chars (%r); max safe prefix is %d. "
        "Using %r to avoid AF_UNIX path length errors (see evenet.utilities.ray_tmpdir).",
        len(current),
        current,
        _MAX_RAY_TMPDIR_PREFIX_LEN,
        short,
    )
    os.environ["RAY_TMPDIR"] = short
