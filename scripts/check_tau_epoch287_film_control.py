"""Strict raw pretrain epoch287 loading and FiLM config CPU parity check."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from evenet.network.body.visible_conditioning import FILM_GAIN_KEYS,diffusion_film_gains
def load_policy(runtime_path, checkpoint_path, normalization_path, device, output_runtime, expected_checkpoint):
    """Strict raw-weight load using only production model constructors."""
    from evenet.control.global_config import Config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    runtime = yaml.safe_load(Path(runtime_path).read_text())
    runtime['options']['Dataset']['normalization_file'] = str(normalization_path)
    Path(output_runtime).write_text(yaml.safe_dump(runtime, sort_keys=False))
    cfg = Config()
    cfg.load_yaml(output_runtime)
    model = build_evenet_on_device(cfg, load_normalization_dict(cfg), device).eval()
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    if any(payload.get(k) != v for k, v in expected_checkpoint.items()):
        raise ValueError('Wrong raw checkpoint epoch/global_step')
    saved = payload['state_dict']
    source = {k.removeprefix('model.'): v for k, v in saved.items()}
    target = model.state_dict()
    if len(source) != len(saved):
        raise ValueError('Checkpoint prefix collision')
    missing = set(target) - set(source)
    extra = set(source) - set(target)
    allowed_training = {'famo.w.' + name for name in
                        ('classification', 'regression', 'assignment', 'generation', 'segmentation')}
    if missing or extra - allowed_training:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra - allowed_training)}')
    for key, value in target.items():
        if value.shape != source[key].shape:
            raise ValueError('Checkpoint shape mismatch: ' + key)
        if 'normalizer.' in key and not torch.equal(value.cpu(), source[key].cpu()):
            raise ValueError('Pinned normalization mismatch: ' + key)
    model.load_state_dict({k: source[k] for k in target}, strict=True)
    if len(target) != 575 or len(model.TruthGeneration.gen_transformer_blocks) != 1:
        raise ValueError('Require original epoch287 575-tensor, one-block architecture')
    model.requires_grad_(False)
    return model, dict(loaded_existing_tensors=len(target), added_tensors=[],
                       ignored_training_state=sorted(extra), checkpoint_identity=expected_checkpoint)


@torch.no_grad()
def sample(model, raw, ids, spec, device, candidates, steps, seed):
    """Production DDIM with fixed per-event noise and actual process labels."""
    import hashlib
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    batch = unpack_event_inputs(torch.as_tensor(raw, device=device), spec)
    label = batch.get('classification')
    if not torch.is_tensor(label) or label.shape not in ((len(raw),), (len(raw), 1)):
        raise ValueError('Cached input must include the actual process labels')
    label = label.reshape(len(raw))
    if (label.dtype == torch.bool or not torch.isfinite(label).all()
            or not torch.equal(label, label.long().to(label.dtype))
            or (label < 0).any() or (label >= 19).any()):
        raise ValueError('Process labels must be integer IDs in [0,18]')
    batch['classification'] = label.long()
    batch['x_invisible'] = torch.zeros(len(raw), 2, 2, device=device)
    batch['x_invisible_mask'] = torch.ones(len(raw), 2, dtype=torch.bool, device=device)
    rows = []
    for identity in ids:
        digest = hashlib.sha256((str(seed) + ':' + str(identity)).encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
        rows.append(torch.randn(candidates, 2, 2, generator=generator))
    noise = torch.stack(rows, dim=1).to(device).reshape(-1, 2, 2)

    class ReplaySampler(DDIMSampler):
        draws_used = 0

        def prior_sde(self, dimensions):
            if self.draws_used or tuple(dimensions) != tuple(noise.shape):
                raise ValueError('Replay noise shape/draw mismatch')
            self.draws_used += 1
            return noise.clone()

    sampler = ReplaySampler(device=device, x0_mode='legacy')
    result = generate_neutrino_candidates(model, batch, sampler, K=candidates,
        num_ddim_steps=steps, device=device, parallel_chains=candidates)
    if sampler.draws_used != 1 or not torch.isfinite(result).all():
        raise ValueError('Invalid replay sampling')
    return result.permute(1, 0, 2, 3).cpu().numpy()


@contextmanager
def reference_gains(model,gains):
    """Independent direct pre-block control for checking configured routing."""
    def hook(module,args,kwargs):
        parts=kwargs['modulation']
        return args,{**kwargs,'modulation':tuple(p if gains[k]==1 else p*gains[k] for p,k in zip(parts,FILM_GAIN_KEYS))}
    handles=[b.register_forward_pre_hook(hook,with_kwargs=True) for b in model.TruthGeneration.gen_transformer_blocks]
    try:yield
    finally:
        for h in handles:h.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('--local-source',type=Path,help='Existing downloaded source.ckpt/normalization.pt/initial_tau_reward.pt')
    p.add_argument('--inputs',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--export-network',type=Path,help='Export the resolved network section for external evaluators')
    a=p.parse_args();r=yaml.safe_load(a.config.read_text());evaluation=r['evaluation']
    if evaluation['expected_checkpoint']!={'epoch':286,'global_step':18655} or evaluation['weights']!='raw':
        raise ValueError('Require pinned original pretrain epoch287 raw weights')
    options=r['network']['VisibleConditioning']
    if options.get('diffusion_enabled') is not True or any(dict(options.get(k,{}) or {}).get('enabled',False) for k in
            ('diffusion_token_readout','diffusion_reward_probe','diffusion_relation_adapter','diffusion_pair_attention','diffusion_met_conditioning','diffusion_leg_attention_conditioning')):
        raise ValueError('Require original epoch287 FiLM architecture; keep module enabled and no later adapters')
    gains=diffusion_film_gains(options.get('diffusion_film_gains'))
    checkpoint=Path(evaluation['checkpoint']);norm=Path(r['options']['Dataset']['normalization_file'])
    bundle_path=checkpoint.parent/'initial_tau_reward.pt'
    if a.local_source:
        checkpoint=a.local_source/'source.ckpt';norm=a.local_source/'normalization.pt';bundle_path=a.local_source/'initial_tau_reward.pt'
    if a.output and a.output.exists():raise ValueError('Preserve previous validation result')
    if a.export_network and a.export_network.exists():raise ValueError('Preserve existing network export')
    panel=a.inputs or Path('/pscratch/sd/y/yiren/Ztautau/tau_dgpo_process_labels_film/run-01/validation_panel.npz')
    bundle=torch.load(bundle_path,map_location='cpu',weights_only=False)
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    spec=EventPackingSpec.from_dict(bundle['head']['packing_spec'])
    with np.load(panel) as z:raw,ids=z['raw_condition'][:2],z['source_ids'][:2]
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    with tempfile.TemporaryDirectory(prefix='epoch287-film-config-') as d:
        root=Path(d);baseline=yaml.safe_load(a.config.read_text())
        baseline['network']['VisibleConditioning'].pop('diffusion_film_gains',None)
        (root/'base.yaml').write_text(yaml.safe_dump(baseline,sort_keys=False))
        original,base_loaded=load_policy(root/'base.yaml',checkpoint,norm,'cpu',root/'base-runtime.yaml',evaluation['expected_checkpoint'])
        controlled,loaded=load_policy(a.config,checkpoint,norm,'cpu',root/'controlled-runtime.yaml',evaluation['expected_checkpoint'])
        assert original.state_dict().keys()==controlled.state_dict().keys()
        for k,v in original.state_dict().items():torch.testing.assert_close(v,controlled.state_dict()[k],rtol=0,atol=0)
        args=(raw,ids,spec,torch.device('cpu'),1,20,6301)
        old=sample(original,*args)
        with reference_gains(original,gains):expected=sample(original,*args)
        actual=sample(controlled,*args)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(old,sample(original,*args))
        report=dict(status='passed',checkpoint=str(checkpoint),identity=evaluation['expected_checkpoint'],
            configured_gains=gains,loaded=loaded,baseline_loaded=base_loaded,all_original_tensors_identical=True,
            exact_ddim20_config_vs_direct_hook_parity=True,events=2,candidates=1,optimizer_steps=0,
            max_abs_offset_change=float(np.max(abs(actual-old))),
            limitation='Config/loading/functional validation, not unfolded-uncertainty performance')
        if a.output:
            a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.export_network:
            a.export_network.parent.mkdir(parents=True,exist_ok=True)
            a.export_network.write_text(yaml.safe_dump(r['network'],sort_keys=False))
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
