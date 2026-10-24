"""Epoch287 DGPO with combined reward, refit10/validation5; user launch only."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from RL.DGPO_neutrino.tau_combined_reward import CONTRACT, validate_combined_head, contract_for_protocol
from scripts.tau_general_spin_angles import PROTOCOL, ROOT_PROTOCOL, branch_spec


def settings(path):
    c=yaml.safe_load(Path(path).read_text())
    branch_spec(protocol=c.get('spin_protocol', PROTOCOL))
    if c.get('wandb_profile') not in (None, 'tau_spin'):
        raise ValueError('Unknown tau-DGPO W&B monitoring profile')
    if c.get('wandb_profile') == 'tau_spin' and c.get('spin_protocol') != ROOT_PROTOCOL:
        raise ValueError('tau_spin monitoring requires the corrected root-exporter geometry')
    if (c['workers'],c['epochs'],c['refit_every_epochs'],c['validation_every_epochs'],c['cij_bootstrap'])!=(16,1500,10,5,1000):
        raise ValueError('Require16GPUs,1500epochs,refit10,validation5,bootstrap1000')
    if not c['run_name'].isascii() or len(c['run_name'])>=96:
        raise ValueError('Readable ASCII run name required')
    if 'classifier_evaluation' in c:
        if (c['classifier_evaluation'] != dict(reweighting_mode='within_event', evaluation_event_weighting='uniform')
                or c.get('spin_protocol') != ROOT_PROTOCOL or c.get('require_equal_event_weights') is not True):
            raise ValueError('Within-event deployment requires corrected geometry and equal event weights')
    if c.get('diffusion_met_conditioning'):
        options=c['diffusion_met_conditioning']
        if (options != dict(enabled=True,width=64,seed=42) or not c.get('classifier_evaluation')
                or c.get('wandb_profile') != 'tau_spin'):
            raise ValueError('MET ablation requires width64, fixed seed42 and corrected within-event source')
    out,src=Path(c['output']).resolve(),Path(c['classifier_directory']).resolve()
    if out.is_relative_to(src) or src.is_relative_to(out):raise ValueError('Separate experiment output required')
    return c


def validate_classifier_evidence(c, source, report):
    """Pin the evaluated head's protocol and diagnostic, not its outcome.

    A user-requested policy-transfer experiment may follow a failed topology
    guard. Preserve that result; never relabel it as a passed selection gate.
    """
    if not report.get('complete'):
        raise ValueError('Require completed combined-classifier fit and held-out scoring')
    expected = c.get('classifier_evaluation')
    if expected is None:
        return
    if any(source.get(k) != v or report['summary'].get(k) != v for k, v in expected.items()):
        raise ValueError('Classifier evaluation weighting differs from the selected experiment')
    if (report['cij']['protocol'] != c['spin_protocol']
            or report['summary'].get('policy_updates') != 0
            or report['cij']['weighting'].get('fixed_event_mass') is not True
            or report['cij']['weighting'].get('event_mass_max_abs_change', float('inf')) > 1e-12):
        raise ValueError('Classifier evidence must preserve event masses at the initial policy')


def validate_equal_event_weights(c, panel):
    if c.get('require_equal_event_weights'):
        w = np.asarray(panel['event_weight'])
        if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or w[0] <= 0 or not np.all(w == w[0]):
            raise ValueError('This experiment requires equal positive event weights; no silent reweighting')


def verify(c):
    from scripts.run_tau_general_spin_angles import settings as classifier_settings, verify as classifier_verify
    from scripts.train_dgpo_tau_process_labels import validate_settings
    from scripts.train_conditional_spin_ratio import build_classifier
    root=Path(c['classifier_directory'])
    s=classifier_settings(root/'settings.json')
    if s['output']!=str(root) or s['wandb_id']!=c['classifier_run']:
        raise ValueError('Classifier source run/output differs')
    classifier_verify(s)
    report = json.loads((root/'report.json').read_text())
    if not (root/'COMPLETE').is_file():
        raise ValueError('Require completed combined-classifier fit and held-out scoring')
    validate_classifier_evidence(c, s, report)
    if c.get('require_equal_event_weights'):
        for name in ('train', 'validation'):
            with np.load(root/(name+'_panel.npz'), allow_pickle=False) as panel:
                validate_equal_event_weights(c, panel)
    head=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    protocol = c.get('spin_protocol', PROTOCOL)
    if (s.get('spin_protocol', PROTOCOL) != protocol
            or head['general_spin_angles']['spin_protocol'] != protocol):
        raise ValueError('Classifier geometry differs; refit a fresh head for this protocol')
    status=json.loads((root/'best.json').read_text())
    if (not status['minimum_fit_steps_met'] or head['optimizer_steps']!=status['best_steps']
            or head['epoch']!=status['best_epoch'] or head['candidate_dim']!=562 or head['condition_dim']!=531):
        raise ValueError('Classifier selection/schema differs')
    head=copy.deepcopy(head)
    head.update(deployment_contract=contract_for_protocol(protocol),source_policy_step=0,fitting_current_policy=True)
    validate_combined_head(head)
    model=build_classifier(head);model.load_state_dict(head['state_dict'],strict=True)
    bundle=torch.load(root/'feature_bundle.pt',map_location='cpu',weights_only=True)
    source=yaml.safe_load((Path(s['source_directory'])/'runtime.yaml').read_text())
    validate_settings(source)
    if c.get('diffusion_met_conditioning'):
        from scripts.tau_met_conditioning import classifier_met_contract
        classifier_met_contract(head, source)
    for key in ('condition_dim','packing_spec','relative_preprocessing','policy_conditioning_contract'):
        if head[key]!=bundle['head'][key]:raise ValueError('Classifier/source schema differs: '+key)
    p=torch.load(bundle['source_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if p.get('dgpo_checkpoint_version') or (p['epoch'],p['global_step'])!=(286,18655):
        raise ValueError('Policy must start at supervised10% epoch287, not later DGPO')
    bundle['head']=head
    bundle['combined_classifier_run']=c['classifier_run']
    if c.get('classifier_evaluation'):
        print('SOURCE EVIDENCE: Cij supported=%s; topology guard=%s; supports_next_policy_test=%s; exploratory policy transfer' % (
            report['summary']['cij/supported_improvement'], report['summary']['topology/point_guard_passed'],
            report['summary']['supports_next_policy_test']), flush=True)
    print('VERIFIED: completed combined head, cap30, raw epoch287, filtered10%, no new fit',flush=True)
    return s,source,bundle


def assemble(source,c):
    r=copy.deepcopy(source);out=Path(c['output']);dg=r['dgpo'];tau=dg['tau_ratio']
    if dg['steps_per_epoch']!=10 or dg['advantage_estimator']!='leave_one_out_unscaled' or dg['K']!=8:
        raise ValueError('Original DGPO update definition changed')
    if dg['reference_trust']['objective']!='velocity_mse' or dg['reference_trust']['coefficient']!=1.:
        raise ValueError('Require original coefficient1 velocity reference penalty')
    if any(tau.get(k) for k in ('fixed_general_round','combined_reward','production_cross_attention')):
        raise ValueError('Require original General runtime as template')
    for key in ('mechanism_probe','transfer_probe'):
        if tau.get(key,{}).get('enabled'):raise ValueError('Cannot inherit another intervention')
    dg.update(checkpoint_load_mode='weights_only',auto_resume_from_last=True,
        auto_resume_fallback_checkpoint_path=None,auto_resume_best_source_checkpoint_dir=None,
        pinned_classifier_restart=False,best_source_start_new_experiment=False,
        step_zero_architecture_bootstrap=False,unbounded_training=False)
    dg['lr_schedule']['resume_use_config']=False
    dg['validation_every_n_epochs']=c['validation_every_epochs']
    dg['validation_full_every_n_epochs']=c['validation_every_epochs']
    # Retain the original15000-update endpoint and LR horizon.
    if dg['lr_schedule'].get('total_epochs')!=1500:raise ValueError('Source LR horizon changed')
    r['options']['Training'].update(epochs=c['epochs'],total_epochs=c['epochs'],
        pretrain_model_load_path=None,model_checkpoint_load_path=str(out/'source.ckpt'),
        model_checkpoint_save_path=str(out/'checkpoints'))
    r['options']['Dataset']['normalization_file']=str(out/'normalization.pt')
    tau.update(output_root=str(out),output=str(out/'tau_diagnostics'),
        train_panel=str(out/'train_panel.npz'),validation_panel=str(out/'validation_panel.npz'),combined_reward=True,
        bootstrap_fresh_classifier=False,validation_every_epochs=c['validation_every_epochs'],
        refit_every_epochs=c['refit_every_epochs'],validation_relative_to_install=False,refit_relative_to_install=False,
        combined_classifier_directory=c['classifier_directory'],cij_bootstrap=c['cij_bootstrap'])
    r['reward_config']=dict(type='conditional_tau',weight=1.,conditional_tau=dict(
        bundle_file=str(out/'initial_tau_reward.pt'),feature_batch_size=256))
    r['logger']['local'].update(name='tau-dgpo-spin-angles',save_dir=str(out/'logs'))
    r['logger']['wandb'].update(id=c['wandb_id'],run_name=c['run_name'],fresh_run=False,resume='allow',
        group='Tau DGPO physics reward',tags=['DGPO','GeneralSpinTau','Refit10','Validation5','Epoch287','Filtered10pct','GPU16','Cap30'])
    if c.get('wandb_profile'):
        r['logger']['wandb']['profile'] = c['wandb_profile']
        tau['wandb_profile'] = c['wandb_profile']
    r['nersc']['ray']['results_dir']=str(out/'ray_results');r['nersc']['submit']=False
    r['experiment']=dict(protocol=contract_for_protocol(c.get('spin_protocol', PROTOCOL)),classifier_run=c['classifier_run'],source_completed_epochs=287,
        source_global_step=18655,stop_policy_step=c['epochs']*10,validation_every_updates=50,refit_every_updates=100,
        reward='Existing combined head bounded log ratio; no within-event softmax in training',
        optimization='Unchanged DGPO leave-one-out and reference velocity MSE coefficient1',
        classifier_refits=True,reference_recenter=True,lr_horizon_steps=15000,
        primary='Native unweighted matched calibrated channel Cij improvement at final endpoint versus0; paired CI95 upper<0',
        audit='Independent fresh unbounded General, selected >=1000 updates; never installed',
        scope='Exploratory conditional generation; no claim of unfolded concurrence closure')
    if c.get('spin_protocol') == ROOT_PROTOCOL:
        r['experiment'].update(spin_protocol=ROOT_PROTOCOL,
            primary='Root exporter target/evenet channel Cij error at final endpoint versus step0; paired CI95 upper<0',
            target='Full MC tau p4 plus reconstructed visible, exactly root exporter target',
            prediction='Offsets plus reconstructed visible then root common.py calibration',
            limitation='Target and prediction use different reconstruction maps; perfect offsets need not close this distance')
        r['logger']['wandb']['tags'].append('RootExporterGeometry')
    if c.get('classifier_evaluation'):
        r['experiment'].update(source_classifier_evaluation=c['classifier_evaluation'],
            source_classifier_report=str(out/'source_classifier_report.json'),
            event_weighting='Equal positive event weights verified in train and validation; common scale preserved',
            deployment_scope='User-requested exploratory policy transfer; source topology guard failed',
            diagnostic_vs_training='Within-event softmax was an offline diagnostic, not the DGPO objective')
        r['logger']['wandb']['tags'].extend(['WithinEventEvidence','EqualEvents','ExploratoryPolicyTransfer'])
    if c.get('diffusion_met_conditioning'):
        from scripts.tau_met_conditioning import global_names
        from evenet.network.body.met_conditioning import met_spec
        spec=met_spec(c['diffusion_met_conditioning'],global_names(r))
        visible=r['network']['VisibleConditioning']
        if visible.get('diffusion_met_conditioning'):
            raise ValueError('Require original epoch287 visible-only FiLM as source')
        visible['diffusion_met_conditioning']=copy.deepcopy(c['diffusion_met_conditioning'])
        r['options']['Training']['model_checkpoint_load_path']=str(out/'policy_source.ckpt')
        r['logger']['wandb']['tags'].append('ObservedMETFiLM')
        r['experiment'].update(met_conditioning=spec,
            classifier_met='Existing General/Spin/Tau heads receive all observed globals, including missing and lead momenta',
            source_function='Zero-output residual preserves all epoch287 velocities at startup',
            question='Does direct MET-related FiLM improve conditional generation and Cij unfolding precision?',
            downstream='Compare saved native candidates locally with step0 response/test split; no in-training unfolding')
    return r


def prepare(c):
    s,source,bundle=verify(c);out=Path(c['output']);marker=out/'experiment_settings.json'
    if marker.exists():
        if json.loads(marker.read_text())!=c:raise ValueError('Experiment settings changed')
        if not (out/'PREPARED').exists():raise ValueError('Incomplete preparation; preserve output and choose run-02')
        path=out/'runtime.yaml';expected=assemble(source,c)
        if yaml.safe_load(path.read_text())!=expected:
            if (out/'checkpoints').exists():raise ValueError('Runtime changed after checkpoint creation; do not alter a resumed experiment')
            path.write_text(yaml.safe_dump(expected,sort_keys=False))
        return path
    if out.exists() and any(out.iterdir()):raise ValueError('Output already exists')
    out.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(c,indent=2)+'\n')
    if c.get('classifier_evaluation'):
        shutil.copy2(Path(c['classifier_directory'])/'report.json', out/'source_classifier_report.json')
    for key,name in [('source_checkpoint','source.ckpt'),('normalization_file','normalization.pt')]:
        shutil.copy2(bundle[key],out/name);bundle[key]=str(out/name)
    runtime=yaml.safe_load(Path(bundle['backbone_runtime']).read_text())
    runtime['options']['Dataset']['normalization_file']=str(out/'normalization.pt')
    (out/'feature_runtime.yaml').write_text(yaml.safe_dump(runtime,sort_keys=False))
    bundle['backbone_runtime']=str(out/'feature_runtime.yaml')
    torch.save(bundle,out/'initial_tau_reward.pt')
    # Preserve old kappas for historical monitors; new monitor explicitly converts.
    from scripts.tau_general_spin_angles import truth_features, validate_truth_cache
    for name,n in [('train',416701),('validation',119002)]:
        with np.load(Path(s['source_directory'])/(name+'_panel.npz'),allow_pickle=False) as f:
            panel={key:f[key] for key in f.files}
        if c.get('spin_protocol') == ROOT_PROTOCOL:
            with np.load(Path(c['classifier_directory'])/(name+'_panel.npz'),allow_pickle=False) as f:
                if str(f['spin_protocol']) != ROOT_PROTOCOL:
                    raise ValueError('Classifier truth p4 cache protocol differs')
                for key in ('source_ids', 'visible_a', 'visible_b', 'truth_deltas', 'event_weight', 'category', 'candidate_truth'):
                    np.testing.assert_array_equal(f[key], panel[key])
                panel.update(truth_tau_a=f['truth_tau_a'], truth_tau_b=f['truth_tau_b'],
                             spin_protocol=np.array(ROOT_PROTOCOL))
        extra=truth_features(panel)
        validate_equal_event_weights(c, panel)
        if panel['candidate_truth'].shape!=(n,533):raise ValueError('Invalid source '+name+' panel')
        panel['candidate_truth']=np.concatenate((panel['candidate_truth'],extra),1)
        validate_truth_cache(panel, bundle['head']['general_spin_angles'])
        np.savez(out/(name+'_panel.npz'),**panel)
    r=assemble(source,c);path=out/'runtime.yaml';path.write_text(yaml.safe_dump(r,sort_keys=False))
    if c.get('diffusion_met_conditioning'):
        from scripts.tau_met_conditioning import prepare_policy_source, classifier_met_contract
        upgrade=prepare_policy_source(path,out/'source.ckpt')
        upgrade['classifier']=classifier_met_contract(bundle['head'],r)
        (out/'met_preparation.json').write_text(json.dumps(upgrade,indent=2)+'\n')
    (out/'PREPARED').write_text('Combined classifier, epoch287 source, cap30, DGPO refit10/validation5; no training started.\n')
    return path


def preflight(path):
    from evenet.control.global_config import global_config as cfg
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device,load_normalization_dict,apply_component_freezes,assert_dgpo_neutrino_policy_deterministic,resolve_dgpo_auto_resume_checkpoint
    from scripts.evaluate_diffusion_pretrain_cij import strict_load
    from RL.DGPO_neutrino.conditional_tau_reward import load_reward
    from RL.DGPO_neutrino.dgpo_trainer import build_optimizer
    from RL.DGPO_neutrino.tau_fixed_round import assert_same_state
    cfg.load_yaml(path);cfg.load_yaml(path)
    resumed=resolve_dgpo_auto_resume_checkpoint(cfg.options.Training.model_checkpoint_save_path,enabled=True)
    source=resumed or cfg.options.Training.model_checkpoint_load_path
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    model=build_evenet_on_device(cfg,load_normalization_dict(cfg),torch.device('cpu'))
    flags={k:p.requires_grad for k,p in model.named_parameters()};strict_load(model,payload)
    # EveNet holds non-picklable config views; construct with the same factory
    # and strictly load weights instead of deep-copying the model object.
    reference=build_evenet_on_device(cfg,load_normalization_dict(cfg),torch.device('cpu'))
    strict_load(reference,payload)
    if resumed:strict_load(reference,{'state_dict':payload['dgpo_ref_state_dict']})
    for k,p in model.named_parameters():p.requires_grad_(flags[k])
    apply_component_freezes(model,cfg);model.train();assert_dgpo_neutrino_policy_deterministic(model)
    import math
    warm=max(1,math.ceil(float(cfg.options.Training.learning_rate_warm_up_factor)*10))
    optimizer=build_optimizer(model,steps_per_epoch=10,warmup_steps=warm,
        lr_schedule=cfg.dgpo['lr_schedule'],conditioning_learning_rates=cfg.dgpo.get('conditioning_learning_rates'))
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer coverage mismatch')
    reward=load_reward(cfg.reward_config.conditional_tau.bundle_file,torch.device('cpu'))
    if resumed:
        if not 0<=payload['global_step']<=15000:raise ValueError('Resume outside experiment range')
        optimizer.load_state_dict(payload['dgpo_optimizer_state_dict'],use_config_lr_schedule=False)
        reward.load_stack_payload(payload['dgpo_omnifold_reward_stack'])
        if reward.installed_head['source_policy_step']!=reward.denominator_step:
            raise ValueError('Resumed reward denominator step mismatch')
    else:
        if getattr(model.TruthGeneration.visible_conditioning,'met_conditioning',None) is not None:
            from evenet.network.body.met_conditioning import assert_zero_met_extension
            assert_zero_met_extension(reference,reward.backbone)
        else:
            assert_same_state(reference.state_dict(),reward.backbone.state_dict(),'initial epoch287 reference')
    if getattr(model.TruthGeneration.visible_conditioning,'met_conditioning',None) is not None:
        from scripts.tau_met_conditioning import preflight_met_inputs
        report=preflight_met_inputs(model,reward,cfg.dgpo['tau_ratio']['validation_panel'],
                                   yaml.safe_load(Path(path).read_text()),initial=not bool(resumed))
        (Path(path).parent/'met_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MET PREFLIGHT: observed globals present in all classifier branches; actor gradient verified',flush=True)
    head_label = 'combined' if reward.installed_head.get('general_spin_angles') else 'General'
    cadence = cfg.dgpo['tau_ratio']
    print(f"PREFLIGHT OK: strict actor/reward/reference weights, optimizer coverage, {head_label} head, "
          f"refit{cadence['refit_every_epochs']}/validation{cadence['validation_every_epochs']}; training not started",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config')
    p.add_argument('phase',choices=['check','verify','prepare','preflight','train'],nargs='?',default='check')
    a=p.parse_args();c=settings(a.config)
    if a.phase=='check':print('CONFIG OK: combined DGPO,epoch287,1500epochs,refit10,validation5,16GPUs');return
    if a.phase=='verify':verify(c);return
    runtime=prepare(c)
    if a.phase in ('preflight','train'):preflight(runtime)
    if a.phase=='train':
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join((str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),env.get('PYTHONPATH','')))
        subprocess.run([sys.executable,'-u',str(ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),str(runtime),
            '--max-steps',str(c['epochs']*10),'--ray-dir',str(Path(c['output'])/'ray_results')],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
