"""Recover source IDs for an archived H4 panel using exact unique visible context.

Reads trusted torch artifacts; no classifier scoring or sample regeneration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

KEYS=[f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('px','py','pz')]


def row_keys(x):
    x=np.asarray(x,dtype='<f4').copy()
    if not np.isfinite(x).all(): raise ValueError('Nonfinite visible context')
    x[x==0]=0  # Canonicalize signed zeros, no rounding/tolerance matching.
    return [r.tobytes() for r in x]


def match_context(wanted,observed):
    wk,ok=row_keys(wanted),row_keys(observed)
    if len(set(wk))!=len(wk): raise ValueError('Candidate context is not unique; need original event-ID sidecar')
    counts={}; indices={}
    needed=set(wk)
    for i,k in enumerate(ok):
        if k in needed:
            counts[k]=counts.get(k,0)+1; indices[k]=i
    if any(counts.get(k,0)!=1 for k in wk):
        raise ValueError('Missing or ambiguous exact context match; never fall back to row order')
    return np.array([indices[k] for k in wk])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Contains fixed_k8_panel.pt and fixed_k1_pool.pt')
    p.add_argument('--weights',type=Path,required=True,help='counterfactual_weights.pt')
    p.add_argument('--arm',required=True,help='Exact saved weight key; use --list-arms to inspect')
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--list-arms',action='store_true')
    args=p.parse_args()
    weights=torch.load(args.weights,map_location='cpu',weights_only=False)
    if args.list_arms:
        print('\n'.join(k for k in weights if k!='valid_event_mask')); return
    report=json.loads((args.weights.parent/'report.json').read_text())
    if Path(report['source_dir']).resolve()!=args.source.resolve():
        raise ValueError('Weight report source_dir does not match --source')
    if not report.get('fixed_candidate_panel') or report.get('candidate_regenerations')!=0:
        raise ValueError('Weights not certified as belonging to the immutable candidate panel')
    panel=torch.load(args.source/'fixed_k8_panel.pt',map_location='cpu',weights_only=False)
    pool=torch.load(args.source/'fixed_k1_pool.pt',map_location='cpu',weights_only=False)
    packed=panel['packed_event'].numpy(); offset=0; fields={}
    for name,shape in pool['packing_spec']['shapes'].items():
        size=int(np.prod(shape)); fields[name]=packed[:,offset:offset+size].reshape(len(packed),*shape); offset+=size
    if offset!=packed.shape[1] or not set(KEYS)<=set(fields): raise ValueError('Packing spec missing visible context or mismatched width')
    mask=weights['valid_event_mask'].numpy().astype(bool)
    if mask.shape!=(len(packed),): raise ValueError('Weights belong to a different panel')
    expected=(panel['policy_noise_mask'].reshape(len(packed),-1)[:,:2]>0).all(1).numpy()
    if not np.array_equal(mask,expected): raise ValueError('Saved weight validity mask differs from source panel')
    w=np.asarray(weights[args.arm],dtype=float)
    d=panel['candidates_kb22'].permute(1,0,2,3).numpy()[mask]
    if w.shape!=d.shape[:2] or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1):
        raise ValueError('Expected saved within-event normalized weights [N,K]')
    cols=KEYS+['source_sample_index','source_event_key']
    files=sorted(args.events.glob('*.parquet')) if args.events.is_dir() else [args.events]
    if not files: raise ValueError(f'No parquet files in {args.events}')
    table=pq.read_table([str(f) for f in files],columns=cols)
    observed=np.stack([table[k].to_numpy() for k in KEYS],-1)
    wanted=np.stack([fields[k].reshape(-1) for k in KEYS],-1)[mask]
    order=match_context(wanted,observed)
    log=np.full_like(w,-np.inf); np.log(w,out=log,where=w>0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,source_sample_index=table['source_sample_index'].to_numpy()[order],
        source_event_key=table['source_event_key'].to_numpy()[order],deltas=d,
        truth_deltas=panel['truth'].reshape(-1,2,2).numpy()[mask],log_ratio=log)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(dict(source=str(args.source.resolve()),weights=str(args.weights.resolve()),arm=args.arm,
        events=str(args.events.resolve()),matching='Exact unique six-component visible context (float32); downstream truth-delta check mandatory',
        weight_mode='conditional',warning='Weights file must originate from this source panel; preserved candidate order, no sorting or regeneration'),indent=2))
    print('READY:',args.output)


if __name__=='__main__': main()
