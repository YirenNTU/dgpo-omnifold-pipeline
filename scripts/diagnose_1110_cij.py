"""Replay h4ratio1 held-out raw ratios on step-1110 samples; no fits or generation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.export_ztautau_cij_candidates import KEYS, match_context
from scripts.diagnose_h4_ratio_tail import unpack
from scripts.diagnose_ztautau_cij import run, read_event_table
from RL.DGPO_neutrino.omnifold_ztautau.logit_calibration import validate_source

BASE = Path('/pscratch/sd/y/yiren/Ztautau')
SOURCE = BASE / 'h4_scale_standardized_long-ratio-health/ratio_audit'
POLICY = BASE / 'dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt'


def export(bundle, table, output):
    """Recover identities exactly; preserve original K=1 physical samples/logits."""
    _, logits, _, _ = validate_source(bundle)
    if not bundle.get('model_state'):
        raise ValueError('Missing saved classifier state')
    fields = unpack(bundle)
    wanted = np.stack([np.asarray(fields[k]).reshape(-1) for k in KEYS], -1)
    observed = np.stack([table[k].to_numpy() for k in KEYS], -1)
    order = match_context(wanted, observed)
    n = len(logits)
    arrays = {}
    for key in ('test_truth', 'test_generated'):
        value = np.asarray(bundle[key])
        if value.shape not in ((n, 4), (n, 2, 2)) or not np.isfinite(value).all():
            raise ValueError(f'Invalid physical K=1 array: {key}')
        arrays[key] = value.reshape(n, 2, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        source_sample_index=table['source_sample_index'].to_numpy()[order],
        source_event_key=table['source_event_key'].to_numpy()[order],
        truth_deltas=arrays['test_truth'], deltas=arrays['test_generated'][:, None],
        log_ratio=logits[:, None])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--events', type=Path, default=BASE/'diffusion_val_20pct_seed42_stic_filtered_test1/val')
    p.add_argument('--output', type=Path, default=BASE/'h4_step1110_cij')
    p.add_argument('--tt2l-repo', type=Path)
    p.add_argument('--kappa-signs', type=int, nargs=2, choices=[-1, 1])
    p.add_argument('--selection')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    if not args.export_only and args.kappa_signs is None:
        p.error('Analysis requires explicit --kappa-signs; or use --export-only')
    if not (args.source/'COMPLETE').is_file():
        raise ValueError('Incomplete ratio export')
    config_path = args.source.parent/'resolved_ratio_experiment.yaml'
    config = yaml.safe_load(config_path.read_text())
    source_policy = config['options']['Training']['model_checkpoint_load_path']
    if str(source_policy) != str(POLICY):
        raise ValueError(f'Not the pinned step-1110 source: {source_policy}')
    bundle = torch.load(args.source/'best_classifier_and_test.pt', map_location='cpu', weights_only=True)
    import pyarrow.parquet as pq
    table = read_event_table(args.events, columns=KEYS+['source_sample_index','source_event_key'])
    args.candidates = args.output/'candidates.npz'
    n = export(bundle, table, args.candidates)
    provenance = dict(policy=source_policy, classifier=str(args.source/'best_classifier_and_test.pt'),
        historical_run='h4ratio1', config=str(config_path), events=n, candidates_per_event=1,
        scoring='Saved independent test logits from restored best-BCE classifier; truth label=1',
        weights='exp(raw logit), globally normalized only; no tempering or event normalization',
        policy_updates=0, classifier_fits=0, candidate_regenerations=0,
        limitation='Historical held-out panel reused for post-hoc physics analysis; not a fresh confirmatory test')
    (args.output/'source_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2), flush=True)
    if not args.export_only:
        args.weight_mode = 'joint'
        args.tolerance = 1e-4
        run(args)


if __name__ == '__main__':
    main()
