"""Small, dependency-light invariants for DGPO truth/pred monitoring."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np


def scheduled_epoch(epoch: int, every_n_epochs: int, *, include_epoch_zero: bool = False) -> bool:
    """Return whether a one-indexed logical-epoch cadence fires at ``epoch``.

    ``epoch`` is the zero-indexed loop counter used by the DGPO trainer.  The
    optional epoch-zero inclusion is useful for train-distribution monitoring,
    where one initial panel is valuable even when the regular cadence is long.
    """
    if include_epoch_zero and int(epoch) == 0:
        return True
    cadence = max(1, int(every_n_epochs))
    return (int(epoch) + 1) % cadence == 0


def validation_schedule_tier(
    epoch: int,
    *,
    cheap_every_n_epochs: int,
    full_every_n_epochs: int,
) -> str | None:
    """Choose exactly one validation tier for an epoch.

    A full validation replaces (rather than duplicates) a coincident cheap
    validation.  The cadences need not divide one another, although production
    configs normally make the full cadence a multiple of the cheap cadence.
    """
    if scheduled_epoch(epoch, full_every_n_epochs):
        return "full"
    if scheduled_epoch(epoch, cheap_every_n_epochs):
        return "cheap"
    return None


def paired_finite_truth_pred(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite event pairs and reject arrays whose row pairing was lost."""
    truth = np.asarray(truth_values, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    if truth.size != pred.size:
        raise ValueError(
            f"{context} arrays must preserve event pairing; "
            f"got truth={truth.size} and pred={pred.size}"
        )
    keep = np.isfinite(truth) & np.isfinite(pred)
    return truth[keep], pred[keep]


def append_reference_prediction_arrays(
    destination: MutableMapping[str, list[np.ndarray]],
    feature_arrays: Mapping[str, np.ndarray],
) -> None:
    """Append ``*_pred`` as ``*_ref`` while deliberately ignoring duplicate truth.

    A reference rollout is evaluated against the same truth rows as the current
    rollout.  Appending its ``*_truth`` entries a second time would double the
    truth length and destroy event-by-event correlations.
    """
    prediction_keys = [key for key in feature_arrays if key.endswith("_pred")]
    if feature_arrays and not prediction_keys:
        raise ValueError("reference feature arrays contain no *_pred entries")
    for pred_key in prediction_keys:
        ref_key = f"{pred_key[:-len('_pred')]}_ref"
        if ref_key not in destination:
            raise KeyError(f"missing reference monitoring destination {ref_key!r}")
        destination[ref_key].append(feature_arrays[pred_key])


def select_truth_pred_by_class(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    class_indices: np.ndarray,
    *,
    class_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one class while enforcing feature/class row alignment."""
    truth = np.asarray(truth_values).reshape(-1)
    pred = np.asarray(pred_values).reshape(-1)
    classes = np.asarray(class_indices, dtype=np.int64).reshape(-1)
    if not (truth.size == pred.size == classes.size):
        raise ValueError(
            "per-class train_dist arrays lost alignment: "
            f"truth={truth.size}, pred={pred.size}, class={classes.size}"
        )
    selected = classes == int(class_id)
    return truth[selected], pred[selected]
