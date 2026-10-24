"""Shared process-label contract for tau actor, reference and classifier."""
import torch

POLICY_CONDITIONING_CONTRACT = 'saved_tau_denominator_label0_v1'
SUPERVISED_POLICY_CONDITIONING_CONTRACT = 'supervised_process_labels_v2'


def canonical_policy_batch(batch, *, contract=SUPERVISED_POLICY_CONDITIONING_CONTRACT, num_classes=19):
    """Preserve supervised process labels; a missing label is an error.

    The explicit legacy branch is solely for replaying archived label0 bundles.
    Process classification is a shared condition, not the truth/generated BCE
    target and not an inferred replacement from event_category.
    """
    n = len(batch['x'])
    if contract == POLICY_CONDITIONING_CONTRACT:
        return {**batch, 'classification': torch.zeros(
            n, device=batch['x'].device, dtype=torch.long)}
    if contract != SUPERVISED_POLICY_CONDITIONING_CONTRACT:
        raise ValueError('Unknown tau policy conditioning contract')
    label = batch.get('classification')
    if not torch.is_tensor(label) or label.shape not in ((n,), (n, 1)):
        raise ValueError('Tau DGPO requires the actual supervised classification per event; no label0 fallback')
    label = label.reshape(n)
    if (label.dtype == torch.bool or not torch.isfinite(label).all()
            or not torch.equal(label, label.long().to(label.dtype))
            or (label < 0).any() or (label >= num_classes).any()):
        raise ValueError(f'Process labels must be integer IDs in [0,{num_classes-1}]')
    return {**batch, 'classification': label.to(device=batch['x'].device, dtype=torch.long)}
