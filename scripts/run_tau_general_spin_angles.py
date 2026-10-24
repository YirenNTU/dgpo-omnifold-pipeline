"""Epoch287 / filtered10% / one combined classifier, user-launched on16 GPUs.

Reuse verified pre-DGPO step-zero samples. No policy construction, generation,
policy optimizer update, allocation or submission. The classifier head trains;
the epoch287 feature trunk remains frozen.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.run_tau_physics_branch import load_panel, merge_candidates
from scripts.tau_general_spin_angles import features, head_config, truth_features, branch_spec, PROTOCOL, ROOT_PROTOCOL
from scripts.train_diffusion_film_10pct import TRAIN, VAL

TRAIN_KEYS = ('condition', 'candidate_truth', 'candidate_generated', 'event_weight', 'split', 'source_ids')


def tihsu_panel(panel):
    """Convert known cache metadata, leaving historical replay inputs untouched.

    Parquet/cache powers carry the decay-mode sign but not the tau+ charge
    sign. tihsu applies that extra minus to leg a. Never change candidate
    features or the old kappas used to undo historical Cij normalization.
    """
    from scripts.tau_tihsu_cij import analyzing_powers
    expected = analyzing_powers(panel['category'])
    stored = np.asarray(panel['kappas'])
    if stored.shape != expected.shape or not np.isfinite(stored).all():
        raise ValueError('Invalid cached analyzing powers')
    if np.allclose(stored, expected, rtol=0, atol=1e-6):
        convention = 'tihsu-charge-signed'
    elif np.allclose(stored, expected * [-1, 1], rtol=0, atol=1e-6):
        convention = 'legacy-decay-mode-signed'
    else:
        raise ValueError('Unrecognized or mixed cached analyzing powers convention')
    return dict(panel, kappas=expected, source_kappas=stored.copy(),
                source_kappa_convention=convention)


def check_prepare_output(out, cfg):
    """Allow retry only for this launcher's early, pre-array snapshot failure."""
    if not out.exists() or not any(out.iterdir()):
        return
    snapshots = {'settings.json', 'source_validation_report.json', 'feature_source.ckpt',
                 'normalization.pt', 'feature_runtime.yaml', 'feature_bundle.pt'}
    saved = json.loads((out/'settings.json').read_text()) if (out/'settings.json').is_file() else None
    if (saved is not None and saved.get('spin_protocol') == 'root-angle-exporter-target-p4-v1'
            and cfg.get('spin_protocol') == ROOT_PROTOCOL):
        # Only the restricted PRE-ARRAY snapshot set below can migrate. No
        # prepared features, classifier weights, STARTED or training state.
        saved = dict(saved, spin_protocol=ROOT_PROTOCOL)
    if (saved != cfg
            or any(p.name not in snapshots or not p.is_file() or p.is_symlink() for p in out.iterdir())):
        raise ValueError('Partial output is not an early snapshot of these settings; choose a new readable run number')
    print('Retrying pre-array preparation with identical settings; no training state exists', flush=True)


def settings(path):
    path = Path(path)
    # JSON scientific notation (e.g.1e-05) is a number; YAML1.1 may read it
    # as a string. Use the actual format for saved experiment settings.
    cfg = json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())
    branch_spec(protocol=cfg.get('spin_protocol', PROTOCOL))
    from scripts.tau_candidate_reweighting import validate_weighting
    validate_weighting(cfg.get('reweighting_mode', 'global'), cfg.get('evaluation_event_weighting', 'dataset'))
    required = dict(cache_policy_step=0, source_completed_epochs=287, source_global_step=18655,
                    workers=16, train_events=416701, validation_events=119002,
                    validation_candidates=8, ratio_cap=30, seed=42,
                    filtered_train=TRAIN, filtered_validation=VAL)
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Require raw epoch287, step0 cache, filtered10%, K8, cap30 and16 GPUs')
    f = cfg['fit']
    if (f['batch_size'] != 1024 or f['epochs'] != 250 or min(f['min_steps'], f['min_selected_steps']) < 1000
            or not 0 < f['min_lr'] <= f['lr'] or cfg['bootstrap'] < 1000):
        raise ValueError('Invalid classifier fitting/selection or bootstrap budget')
    src, out = Path(cfg['source_directory']).resolve(), Path(cfg['output']).resolve()
    if out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('Output must be separate from source')
    name = cfg['logger']['name']
    if not name.isascii() or len(name)>96 or len(name.split(' | '))<3:
        raise ValueError('Need a readable ASCII W&B name')
    return cfg


def verify(cfg):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from RL.DGPO_neutrino.conditional_tau_reward import validate_head
    root = Path(cfg['source_directory'])
    runtime = yaml.safe_load((root/'runtime.yaml').read_text())
    for key, directory, count in [('data_parquet_dir', TRAIN, 416701), ('data_parquet_val_dir', VAL, 119002)]:
        if runtime['platform'][key] != directory or validate_filtered_dataset(directory)['rows'] != count:
            raise ValueError('Wrong/incomplete filtered manifest; no raw fallback')
    tau = runtime['dgpo']['tau_ratio']
    if (tau['source_completed_epochs'], tau['source_global_step'], tau['policy_conditioning_contract']) != (
            287, 18655, 'supervised_process_labels_v2'):
        raise ValueError('Wrong source epoch or label conditioning')
    bundle = torch.load(root/'initial_tau_reward.pt', map_location='cpu', weights_only=True)
    ckpt = torch.load(bundle['source_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if (ckpt.get('dgpo_checkpoint_version') or ckpt['epoch'] != 286 or ckpt['global_step'] != 18655
            or not any('visible_conditioning.' in k for k in ckpt['state_dict'])
            or not any(k.endswith('TruthGeneration.label_dense.weight') for k in ckpt['state_dict'])):
        raise ValueError('Frozen feature source must be raw FiLM epoch287')
    del ckpt
    training = root/'tau_diagnostics/refit-step-00000000'
    validation = root/'tau_diagnostics/eval-step-00000000'
    report = json.loads((validation/'report.json').read_text())
    if (report['policy_step'], report['epoch'], report['events'], report['candidates']) != (0, -1, 119002, 8):
        raise ValueError('Require pre-update epoch287 validation, not an RL checkpoint')
    head = torch.load(training/'best.pt', map_location='cpu', weights_only=True)
    validate_head(head)
    if (head['source_policy_step'], head['candidate_dim'], head['condition_dim'], head['ratio_bound']) != (0, 533, 531, 30):
        raise ValueError('Wrong step-zero classifier schema')
    if head.get('num_process_classes') != 19 or head.get('dropout') != 0.:
        raise ValueError('Wrong source process-label vocabulary or dropout')
    for key in ('packing_spec', 'relative_preprocessing', 'policy_conditioning_contract'):
        if head[key] != bundle['head'][key]:
            raise ValueError('Frozen feature schema mismatch: '+key)
    for directory in (training, validation):
        for rank in range(16):
            if not (directory/f'candidates/rank-{rank:02d}.npz').is_file():
                raise ValueError('Incomplete step-zero cache; no substitution from later steps')
    print('SOURCE OK: epoch287 FiLM, pre-update caches, filtered10%, labels and cap30', flush=True)
    return root, training, validation, head, report, bundle


def prepare(cfg):
    root, training, validation, head, reference, bundle = verify(cfg)
    out = Path(cfg['output'])
    if (out/'PREPARED').exists():
        if json.loads((out/'settings.json').read_text()) != cfg:
            raise ValueError('Prepared settings differ')
        return out
    check_prepare_output(out, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out/'settings.json').write_text(json.dumps(cfg, indent=2)+'\n')
    shutil.copy2(validation/'report.json', out/'source_validation_report.json')
    for key, name in [('source_checkpoint','feature_source.ckpt'), ('normalization_file','normalization.pt')]:
        shutil.copy2(bundle[key], out/name); bundle[key] = str(out/name)
    source = yaml.safe_load(Path(bundle['backbone_runtime']).read_text())
    source['options']['Dataset']['normalization_file'] = str(out/'normalization.pt')
    (out/'feature_runtime.yaml').write_text(yaml.safe_dump(source, sort_keys=False))
    bundle['backbone_runtime'] = str(out/'feature_runtime.yaml')
    torch.save(bundle, out/'feature_bundle.pt')
    panels = {key: load_panel(root/(key+'_panel.npz')) for key in ('train','validation')}
    if np.intersect1d(panels['train']['source_ids'],panels['validation']['source_ids']).size:
        raise ValueError('Train/validation event overlap')
    from RL.DGPO_neutrino.conditional_tau_reward import condition_inputs
    from scripts.tau_physics_branch_report import terms
    for label, n, k, folder in [('train',416701,1,training),('validation',119002,8,validation)]:
        p = panels[label]
        if (len(p['source_ids']) != n or len(np.unique(p['source_ids'])) != n
                or not bool(p['truth_features_ready']) or p['candidate_truth'].shape != (n,533)):
            raise ValueError('Incomplete source panel')
        if any(not np.isfinite(p[key]).all() for key in
               ('raw_condition','condition','candidate_truth','truth_deltas','event_weight','visible_a','visible_b')):
            raise ValueError('Nonfinite source panel; no event dropping')
        if (p['event_weight'] < 0).any() or p['event_weight'].sum() <= 0:
            raise ValueError('Invalid source event weights')
        protocol = cfg.get('spin_protocol', PROTOCOL)
        if protocol == ROOT_PROTOCOL:
            from scripts.tau_root_export_geometry import attach_truth_p4
            p = attach_truth_p4(p, cfg['filtered_train' if label == 'train' else 'filtered_validation'])
        calibrated = tihsu_panel(p)
        print(label+' cache analyzing powers: '+calibrated['source_kappa_convention']+
              ' -> tihsu-charge-signed for new Cij reports', flush=True)
        for start in range(0,n,2048):
            sl = slice(start,start+2048)
            c = condition_inputs(p['raw_condition'][sl],p['category'][sl],bundle['condition_mean'],bundle['condition_scale'],head['packing_spec'])
            np.testing.assert_array_equal(c,p['condition'][sl])
        d, f = merge_candidates(folder/'candidates',n,k)
        if protocol == ROOT_PROTOCOL:
            from scripts.tau_root_export_geometry import angular_domain_report
            domain = angular_domain_report(p['visible_a'][:,None], p['visible_b'][:,None], d)
            print(label+' angular-domain audit: '+json.dumps(domain), flush=True)
        if label == 'train':
            if tuple(np.count_nonzero(p['split']==j) for j in (0,1)) != (354488,62213):
                raise ValueError('Original internal split changed')
            a = dict(condition=p['condition'], event_weight=p['event_weight'], split=p['split'], source_ids=p['source_ids'],
                     candidate_truth=np.c_[p['candidate_truth'],truth_features(p)],
                     candidate_generated=np.c_[f,features(p['visible_a'],p['visible_b'],d[:,0],protocol=protocol)])
            for key in TRAIN_KEYS:
                np.save(out/('train_'+key+'.npy'),a[key])
            if protocol == ROOT_PROTOCOL:
                np.savez(out/'train_panel.npz', **p)
        else:
            t = terms(p,p['truth_deltas']); g = np.mean([terms(p,d[:,j]) for j in range(k)],axis=0)
            for value,key in [(t,'truth'),(g,'generated')]:
                np.testing.assert_allclose(np.average(value,weights=p['event_weight'],axis=0).reshape(3,3),reference[key],rtol=1e-7,atol=1e-7)
            with np.load(validation/'measurements.npz',allow_pickle=False) as m:
                np.testing.assert_array_equal(m['source_ids'],p['source_ids'])
                np.testing.assert_array_equal(m['deltas'],d)
                np.testing.assert_array_equal(m['weight'],p['event_weight'])
            np.save(out/'validation_deltas.npy',d)
            np.save(out/'validation_candidate0.npy',f)
            np.savez(out/'validation_panel.npz',**calibrated)
            # Full MC tau + MC visible remains diagnostic-only. The corrected
            # positive Spin branch uses full MC tau + RECO visible instead.
            from scripts.run_tau_general_topology_residual import complete_truth_terms
            # Undo the OLD normalization with OLD powers before the new report
            # applies its charge-signed powers. Substituting calibrated here
            # would flip the full-truth products incorrectly.
            full = complete_truth_terms(p).reshape(-1,3,3) * p['kappas'].prod(1)[:,None,None] / 9
            np.save(out/'full_truth_products.npy', full[:,[2,1,0]][:,:,[2,1,0]].reshape(-1,9))
        if protocol == ROOT_PROTOCOL:
            (out/(label+'_angular_domain.json')).write_text(json.dumps(domain,indent=2)+'\n')
        print('PREPARED '+label+': identities, conditioning and candidate geometry verified',flush=True)
    base = copy.deepcopy(head)
    for key in ('state_dict','epoch','optimizer_steps','val_bce'):
        base.pop(key,None)
    base.update(cfg['fit']); base.update(seed=cfg['seed'])
    torch.save(head_config(base, cfg.get('spin_protocol', PROTOCOL)),out/'head_config.pt')
    (out/'PREPARED').write_text('Fresh combined head; frozen epoch287 trunk; step0 saved candidates; no policy updates.\n')
    return out


def worker(cfg):
    import ray.train
    from ray.train.torch import get_device
    from RL.DGPO_neutrino.conditional_tau_cycle import fit_head, panel_batch, barrier
    from RL.DGPO_neutrino.conditional_tau_reward import load_reward, feature_precision
    from scripts.train_conditional_spin_ratio import build_classifier, pair_metrics
    ctx = ray.train.get_context(); rank = ctx.get_world_rank(); world = ctx.get_world_size(); device = get_device()
    if world != 16 or device.type != 'cuda':
        raise ValueError('Requires existing16-GPU allocation')
    torch.set_num_threads(1)
    out = Path(cfg['output']); hcfg = torch.load(out/'head_config.pt',map_location='cpu',weights_only=True)
    arrays = {k:np.load(out/('train_'+k+'.npy'),mmap_mode='r') for k in TRAIN_KEYS}
    run = None
    if rank == 0:
        import wandb
        run = wandb.init(**cfg['logger'],id=cfg['wandb_id'],resume='never',dir=str(out),
            config={**cfg,'policy_updates':0,'candidate_generation':False,'arms':['combined'],
                    'fit_event_weighting':'dataset','heldout_classifier_event_weighting':'dataset',
                    'head_initialization':'fresh','feature_trunk':'frozen supervised epoch287'})
        run.define_metric('fit/*',step_metric='fit/optimizer_steps')
        (out/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url)))
    try:
        # Check cached fit features against the frozen source before spending
        # any classifier updates, not only when scoring the external panel.
        trunk = load_reward(out/'feature_bundle.pt',device)
        fit_panel = load_panel(Path(cfg['source_directory'])/'train_panel.npz')
        ii = np.arange(rank, min(len(fit_panel['source_ids']),world*8),world)
        with torch.no_grad(),feature_precision():
            b = panel_batch(fit_panel,ii,trunk.spec,device)
            cc,tf = trunk.features(b,torch.as_tensor(fit_panel['truth_deltas'][ii],device=device))
            np.testing.assert_allclose(cc,fit_panel['condition'][ii],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(tf,fit_panel['candidate_truth'][ii],rtol=1e-4,atol=1e-3)
            shard=Path(cfg['source_directory'])/f'tau_diagnostics/refit-step-00000000/candidates/rank-{rank:02d}.npz'
            with np.load(shard,allow_pickle=False) as f:
                _,gf = trunk.features(b,torch.as_tensor(f['deltas'][:len(ii),0],device=device))
                np.testing.assert_allclose(gf,f['features'][:len(ii)],rtol=1e-4,atol=1e-3)
        del fit_panel
        barrier(world)
        def emit(row):
            if run is not None: run.log({'fit/'+k:v for k,v in row.items()})
            print(f"[combined] epoch={row['epoch']} updates={row['optimizer_steps']} val_bce={row['val_bce']:.6f}",flush=True)
        with feature_precision():
            model, saved, status = fit_head(hcfg,arrays,out/'best.pt',device,rank,world,emit)
        del arrays
        if saved['optimizer_steps'] < 1000: raise ValueError('Undertrained selected head')
        if rank == 0: run.summary.update({'fit/'+k:v for k,v in status.items()})
        panel = load_panel(out/'validation_panel.npz')
        d = np.load(out/'validation_deltas.npy',mmap_mode='r')
        candidate0 = np.load(out/'validation_candidate0.npy',mmap_mode='r')
        idx = np.arange(rank,len(d),world)
        scores = np.empty((len(idx),8),np.float32); truth = np.empty(len(idx),np.float32)
        with torch.no_grad(),feature_precision():
            for start in range(0,len(idx),256):
                ii = idx[start:start+256]; b = panel_batch(panel,ii,trunk.spec,device)
                c = panel['condition'][ii]; ct = torch.as_tensor(c,device=device)
                # Replay truth through the frozen trunk as well as negatives.
                cc, tf = trunk.features(b,torch.as_tensor(panel['truth_deltas'][ii],device=device))
                np.testing.assert_allclose(cc,c,rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(tf,panel['candidate_truth'][ii],rtol=1e-4,atol=1e-3)
                x = np.c_[tf,truth_features(panel, ii)]
                truth[start:start+len(ii)] = model(ct,torch.as_tensor(x,device=device)).cpu().numpy()
                for j in range(8):
                    draws = np.array(d[ii,j]); cc, f = trunk.features(b,torch.as_tensor(draws,device=device))
                    np.testing.assert_allclose(cc,c,rtol=1e-6,atol=1e-6)
                    if j==0: np.testing.assert_allclose(f,candidate0[ii],rtol=1e-4,atol=1e-3)
                    x = np.c_[f,features(panel['visible_a'][ii],panel['visible_b'][ii],draws,
                                         protocol=hcfg['general_spin_angles']['spin_protocol'])]
                    scores[start:start+len(ii),j] = model(ct,torch.as_tensor(x,device=device)).cpu().numpy()
                if rank==0: run.log({'scoring/events_per_rank':start+len(ii)})
        np.savez(out/f'scores-rank-{rank:02d}.npz',positions=idx,source_ids=panel['source_ids'][idx],log_ratio=scores,truth_logits=truth)
        barrier(world)
        if rank == 0:
            all_scores = np.empty((len(d),8),np.float32); all_truth = np.empty(len(d),np.float32); seen = np.zeros(len(d),int)
            for rr in range(world):
                with np.load(out/f'scores-rank-{rr:02d}.npz',allow_pickle=False) as f:
                    ii = f['positions']; np.add.at(seen,ii,1)
                    np.testing.assert_array_equal(f['source_ids'],panel['source_ids'][ii])
                    all_scores[ii] = f['log_ratio']; all_truth[ii] = f['truth_logits']
            if not np.all(seen==1): raise ValueError('Missing/duplicate scoring events')
            np.savez(out/'scores.npz',source_ids=panel['source_ids'],log_ratio=all_scores,truth_logits=all_truth)
            from scripts.tau_general_spin_angles_report import write_report
            report, plots = write_report(out,panel,d,all_scores,np.load(out/'full_truth_products.npy'),
                cfg['bootstrap'],cfg['bootstrap_seed'],progress=lambda i:run.log({'report/bootstrap':i}),
                reweighting_mode=cfg.get('reweighting_mode','global'),
                event_weighting=cfg.get('evaluation_event_weighting','dataset'))
            test = pair_metrics(all_truth,all_scores[:,0],panel['event_weight'])
            (out/'heldout_classifier.json').write_text(json.dumps(test,indent=2)+'\n')
            run.summary.update({'heldout/'+k:v for k,v in test.items()})
            run.summary.update(report['summary'])
            run.log({'plots/'+p.stem:wandb.Image(str(p)) for p in plots})
            for file in ('report.json','histograms.json','heldout_classifier.json','settings.json'):
                run.save(str(out/file),base_path=str(out),policy='now')
            (out/'COMPLETE').write_text('Combined classifier and held-out reweighting complete; no DGPO.\n')
        barrier(world)
        ray.train.report({'completed':1})
    finally:
        if run is not None: run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('phase',choices=('check','verify','prepare','run'),nargs='?',default='check')
    parser.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = parser.parse_args(); cfg = settings(args.config)
    if args.phase=='check':
        print('CONFIG OK: one combined classifier, epoch287, filtered10%,16 GPUs, cap30; '
              f"reweighting={cfg.get('reweighting_mode','global')}, "
              f"evaluation events={cfg.get('evaluation_event_weighting','dataset')}; no DGPO")
        return
    if args.phase=='verify': verify(cfg); return
    out = prepare(cfg)
    if args.phase=='prepare': print(out); return
    if (out/'STARTED').exists(): raise ValueError('Already started; use a new output and W&B run number')
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=args.ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(map(str,(ROOT,ROOT/'scripts',ROOT/'evenet_dgpo')))}})
    try:
        if ray.cluster_resources().get('GPU',0)<16: raise ValueError('Requires existing16-GPU Ray allocation')
        with (out/'STARTED').open('x') as started:
            started.write('User launched classifier-only fit and scoring.\n')
        TorchTrainer(worker,train_loop_config=cfg,scaling_config=ScalingConfig(num_workers=16,use_gpu=True),
            run_config=RunConfig(name='general-spin-tau-epoch287',storage_path=str(out/'ray_results'),
                                failure_config=FailureConfig(max_failures=0))).fit()
    finally: ray.shutdown()


if __name__=='__main__': main()
