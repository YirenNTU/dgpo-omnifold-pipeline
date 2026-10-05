"""Fresh K=1 training negatives, with frozen raw policy and frozen preprocessing."""
from contextlib import contextmanager
import time

import numpy as np
import torch


def draw_seed(base, epoch, rank, batch):
    if min(epoch, rank, batch) < 0 or rank >= 16 or batch >= 10000:
        raise ValueError('Invalid fresh-negative stream index')
    return int(base + 1000003*epoch + 10000*rank + batch)


@contextmanager
def isolated_rng(device, seed):
    """Policy construction and sampling must not change head/dropout RNG or precision."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    precision = torch.get_float32_matmul_precision()
    with torch.random.fork_rng(devices=devices):
        # Seed only the CPU and assigned CUDA generator, not all visible devices.
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(precision)


def load_policy(cfg, device):
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from scripts.diagnose_h4_spike_coverage import load_raw_state
    global_config.load_yaml(cfg['fresh_runtime'])
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
    saved = torch.load(cfg['fresh_generator_checkpoint'], map_location='cpu', weights_only=False)
    if int(saved['global_step']) != 1110:
        raise ValueError('Expected raw step1110 generator')
    load_raw_state(policy, saved, 'step1110')
    policy.requires_grad_(False)
    return policy


def make_batch(raw, spec, device):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
    batch = unpack_event_inputs(torch.as_tensor(raw, device=device), spec)
    batch.pop('classification', None)
    batch['x_invisible'] = torch.zeros(len(raw), 2, 2, device=device)
    batch['x_invisible_mask'] = torch.ones(len(raw), 2, dtype=torch.bool, device=device)
    return batch


@torch.no_grad()
def candidate_features(policy, batch, delta, va, vb, stats, microbatch):
    from scripts.tau_backbone_alignment import candidate_hidden
    from scripts.diagnose_ztautau_cij import tau_from_deltas
    from scripts.train_conditional_spin_ratio import tau_features
    from scripts.tau_relative_inputs import relative_angles
    torch.set_float32_matmul_precision('highest')
    hidden = []
    for start in range(0, len(delta), microbatch):
        sub = {k:v[start:start+microbatch] if torch.is_tensor(v) and v.ndim and len(v)==len(delta) else v
               for k,v in batch.items()}
        hidden.append(candidate_hidden(policy, sub, delta[start:start+microbatch]))
    d = delta.cpu().numpy()
    a, b = tau_from_deltas(va,d[:,0]), tau_from_deltas(vb,d[:,1])
    relative = (relative_angles(a,b,va,vb)-np.asarray(stats['mean']))/np.asarray(stats['scale'])
    features = np.concatenate((np.concatenate(hidden), relative, tau_features(a,b)), axis=1).astype('float32')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite fresh candidate features')
    return features


class FreshNegatives:
    def __init__(self, cfg, device, rank, old_features):
        from evenet.utilities.diffusion_sampler import DDIMSampler
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
        self.cfg, self.device, self.rank = cfg, device, rank
        with np.load(cfg['fresh_inputs'], allow_pickle=False) as f:
            self.a = {k:f[k] for k in f.files}
        if len(self.a['raw_condition']) != len(old_features):
            raise ValueError('Fresh inputs must contain only fit rows, in fit order')
        with np.load(cfg['prepared'], allow_pickle=False) as f:
            if not np.array_equal(self.a['source_ids'], f['source_ids'][f['split']==0]):
                raise ValueError('Fresh training identities differ')
        self.spec = EventPackingSpec.from_dict(cfg['packing_spec'])
        with isolated_rng(device, cfg['fresh_seed']):
            self.policy = load_policy(cfg, device)
            self.sampler = DDIMSampler(device=device)
            idx = np.arange(rank, min(len(old_features), 64*16), 16)
            batch = make_batch(self.a['raw_condition'][idx], self.spec, device)
            replay = self.features(batch, torch.as_tensor(self.a['old_deltas'][idx], device=device), idx)
        self.parity = float(np.max(np.abs(replay-old_features[idx])))
        if not np.allclose(replay, old_features[idx], atol=1e-3, rtol=1e-4):
            raise ValueError(f'Frozen feature replay failed on rank {rank}: {self.parity}')
        # Match sample_conditional_spin_train.worker, which used PyTorch's
        # default highest precision, NOT the unused dgpo runtime precision field.
        self.precision = cfg['fresh_generation_precision']

    def features(self, batch, delta, idx):
        return candidate_features(self.policy, batch, delta, self.a['visible_a'][idx],
            self.a['visible_b'][idx], self.cfg['relative_preprocessing'], self.cfg['feature_batch_size'])

    def begin_epoch(self, epoch):
        self.events, self.seconds = 0, 0.

    @torch.no_grad()
    def draw(self, idx, epoch, batch_index):
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        start = time.perf_counter()
        with isolated_rng(self.device, draw_seed(self.cfg['fresh_seed'], epoch, self.rank, batch_index)):
            self.policy.eval()
            batch = make_batch(self.a['raw_condition'][idx], self.spec, self.device)
            torch.set_float32_matmul_precision(self.precision)
            delta = generate_neutrino_candidates(self.policy, batch, self.sampler, K=1,
                num_ddim_steps=20, device=self.device, parallel_chains=1)[0]
            result = torch.as_tensor(self.features(batch, delta, idx), device=self.device)
        self.events += len(idx)
        self.seconds += time.perf_counter()-start
        return result
