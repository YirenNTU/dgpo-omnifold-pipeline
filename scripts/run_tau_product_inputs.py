"""Add nine analyzer outer products to the completed Geometry control;16 GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml

from scripts import run_tau_explicit_inputs as explicit
from scripts.diagnose_tau_weight_factor_swap import PROTOCOL_KEYS
from scripts.run_tau_ratio_objectives import run_arm
from scripts.tau_explicit_inputs import prepare_arrays
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.tau_conditioning_heads import parameter_count
from scripts.tau_tail_attribution import AXES


def read_settings(path):
    overlay = yaml.safe_load(Path(path).read_text())
    settings = explicit.read_settings(ROOT/overlay['explicit_config'])
    logger = dict(settings['logger'], **overlay.pop('logger'))
    settings.update(overlay); settings['logger'] = logger
    if (settings['workers'], settings['batch_size'], settings['ratio_bound'], settings['geometry_run']) != (16,1024,30,'n3jjqcn7'):
        raise ValueError('Requires fixed Geometry control, bound30 and16 GPUs x1024')
    name = logger['products_name']
    if len(name) > 96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid product-ablation display name')
    return settings


def make_config(settings, baseline, source, output, arm):
    if arm != 'products': raise ValueError('This launcher only trains the product intervention')
    cfg = explicit.make_config(settings, baseline, source, output, arm)
    cfg.update(geometry_control=settings['geometry_control'], geometry_run=settings['geometry_run'],
        intervention='Add9 unscaled candidate-owned analyzer outer products to saved Geometry input design; all other inputs unchanged.',
        primary='K64 Cij Frobenius error: products minus saved Geometry n3jjqcn7; paired event bootstrap.',
        secondary='All9 component errors, K1 closure, ESS, category x pT closure; context256 zrv2yfgt is a secondary comparator.',
        initialization='Fresh seed42, shared Geometry initial parameters/function/RNG; only9 new product columns zero.',
        additional_candidate_inputs=9, product_target_leakage=False, Cij_loss=False,
        no_control_refits=True, no_product_cap_sweep=True)
    return cfg


def verify_geometry(settings, baseline, source, original, va, vb, gt, gg):
    directory = Path(settings['geometry_control'])
    if not (directory/'COMPLETE').is_file(): raise ValueError('Geometry control incomplete')
    if json.loads((directory/'wandb.json').read_text())['id'] != settings['geometry_run']:
        raise ValueError('Wrong Geometry run')
    arrays, spec = prepare_arrays(original, va, vb, gt, gg, 'geometry')
    local = dict(settings, explicit_input=spec, input_inventory={}, explicit_parameter_counts={},
        input_dimensions={'condition':arrays['condition'].shape[1], 'candidate':arrays['candidate_truth'].shape[1]})
    expected = explicit.make_config(local, baseline, source, directory, 'geometry')
    cfg = json.loads((directory/'manifest.json').read_text())
    keys = PROTOCOL_KEYS + ('condition_dim','candidate_dim','diagnostic_every','conditioning_diagnostics',
        'skip_train_mmd_diagnostic','panel_directory','wide_directory','wide_run','feature_batch_size','bootstrap')
    for key in keys:
        if key not in cfg or cfg[key] != expected.get(key):
            raise ValueError('Geometry training protocol differs: '+key)
    if cfg['explicit_input'] != spec: raise ValueError('Geometry feature convention differs')
    with np.load(directory/'prepared.npz', allow_pickle=False) as f:
        if set(f.files) != set(arrays) or any(not np.array_equal(f[k], v) for k,v in arrays.items()):
            raise ValueError('Geometry paired data/splits differ')
    checkpoint = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
    for key in ('explicit_input','condition_dim','candidate_dim','condition_width','condition_hidden','ratio_bound'):
        if checkpoint.get(key) != cfg[key]: raise ValueError('Geometry checkpoint differs: '+key)
    model = build_classifier(checkpoint); model.load_state_dict(checkpoint['state_dict'], strict=True)
    _, _, report = explicit.load_saved_scores(directory, arrays['source_ids'][arrays['split']==2])
    explicit.bounded.verify_panel_report(report, Path(settings['panel_directory']))
    return arrays, cfg


def prepare(settings):
    source, baseline, original, scores, obs = explicit.prepare(settings)
    va, vb, gt, gg, inventory = obs
    geometry, control = verify_geometry(settings, baseline, source, original, va, vb, gt, gg)
    arrays, spec = prepare_arrays(original, va, vb, gt, gg, 'products')
    # Enforce "only9 added" across every event, split and cached feature.
    for key, value in arrays.items():
        reduced = np.concatenate((value[:,:-30], value[:,-21:]), axis=1) if key.startswith('candidate_') else value
        if not np.array_equal(reduced, geometry[key]): raise ValueError('Unexpected input intervention: '+key)
    settings.update(explicit_input=spec, input_inventory=inventory, explicit_parameter_counts={},
        input_dimensions={'condition':arrays['condition'].shape[1], 'candidate':arrays['candidate_truth'].shape[1]})
    cfg = make_config(settings, baseline, source, Path('/unused'), 'products')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed']); new = build_classifier(cfg)
        torch.manual_seed(control['seed']); old = build_classifier(control)
        new.eval(); old.eval()
        take = np.flatnonzero(arrays['split']==0)[:32]
        c = torch.from_numpy(arrays['condition'][take])
        t = torch.from_numpy(arrays['candidate_truth'][take])
        t_old = torch.from_numpy(geometry['candidate_truth'][take])
        with torch.no_grad():
            torch.testing.assert_close(new(c,t), old(c,t_old), rtol=1e-6, atol=1e-7)
        settings['explicit_parameter_counts'] = {'products':parameter_count(new), 'geometry':parameter_count(old)}
    print('READY:', dict(new_fits=['products'], control_refits=0, workers=16, batch_size=1024,
        splits=baseline['split_counts'], additional_features=9, ratio_bound=30), flush=True)
    return source, baseline, arrays, scores


def finish(cfg, arrays, p, q, scores, run, settings):
    explicit.finish(cfg, arrays, p, q, scores, run, settings)
    import wandb
    report = json.loads((Path(cfg['output'])/'fixed_K64_report.json').read_text())
    groups = json.loads((Path(cfg['output'])/'fixed_K64_groups.json').read_text())['groups']
    for arm in ('bounded','geometry','condition256','unweighted'):
        error = np.asarray(report['arms'][arm]['absolute_component_error'])
        gr = [r for r in groups if r['arm']==arm and r['group']!='all']
        run.summary.update({f'K64/{arm}/diagonal_error':float(np.linalg.norm(error[[0,4,8]])),
            f'K64/{arm}/offdiagonal_error':float(np.linalg.norm(error[[1,2,3,5,6,7]])),
            f'K64/{arm}/group_mass_tv':.5*sum(abs(r['weighted_mass']-r['base_mass']) for r in gr),
            f'K64/{arm}/base_weighted_group_error':sum(r['base_mass']*r['error'] for r in gr)})
    rows = []
    for control in ('geometry','condition256'):
        contrast = report['comparisons']['bounded_minus_'+control]
        new = np.array(report['arms']['bounded']['absolute_component_error'])
        old = np.array(report['arms'][control]['absolute_component_error'])
        bands = np.array(contrast['absolute_component_error_change_ci95'])
        rows += [[control, name, float(new[j]-old[j]), float(bands[0,j]), float(bands[1,j])]
                 for j,name in enumerate(AXES)]
    run.log({'K64/product_component_changes':wandb.Table(
        columns=['control','component','absolute_error_change','pointwise_lo95','pointwise_hi95'],data=rows)})
    run.summary.update(dict(primary_comparator='geometry', primary_control_run=settings['geometry_run'],
        additional_candidate_inputs=9, Cij_loss=False, control_refits=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('phase', choices=('prepare','train'), nargs='?', default='train')
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = parser.parse_args(); settings = read_settings(args.config)
    source, baseline, arrays, scores = prepare(settings)
    if args.phase == 'train':
        run_arm(settings, 'products', source, baseline, arrays, scores, args.ray_address,
            config_builder=make_config, reporter=finish, materialize_inputs=True)


if __name__ == '__main__': main()
