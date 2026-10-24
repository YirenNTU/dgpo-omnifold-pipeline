"""Exact architecture and normalization loading for supervised continuation."""
import torch


def load_exact_pretrained(model, checkpoint):
    source = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    if len(source) != len(checkpoint['state_dict']):
        raise ValueError('Checkpoint prefix collision')
    target = model.state_dict()
    missing = set(target)-set(source)
    extra = {k for k in source if k not in target and not k.startswith('famo.w.')}
    shapes = [k for k in target if k in source and target[k].shape != source[k].shape]
    if missing or extra or shapes:
        raise ValueError(f'Exact pretrained mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shapes={shapes}')
    for k, v in target.items():
        if 'normalizer.' in k and not torch.equal(v.cpu(), source[k].cpu()):
            raise ValueError('Pinned normalization disagrees with checkpoint: '+k)
    model.load_state_dict({k: source[k] for k in target}, strict=True)
    return {'loaded_tensors': len(target), 'ignored_training_state': sorted(set(source)-set(target))}
