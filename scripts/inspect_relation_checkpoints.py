#!/usr/bin/env python3
"""Read-only checkpoint inventory: no model, GPU, Ray, W&B or training."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/pscratch/sd/y/yiren/Ztautau/diffusion_relation_relations_01'))
    args=p.parse_args()
    found=[];seen=set()
    for candidate in sorted(args.root.rglob('*.ckpt')):
        try:
            path=candidate.resolve(strict=True)
            if path in seen:continue
            seen.add(path)
            ckpt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            epoch=ckpt.get('epoch');state=ckpt.get('state_dict',{})
            row=dict(path=str(path),epoch=epoch,completed_epochs=epoch+1 if isinstance(epoch,int) else None,
                     global_step=ckpt.get('global_step'),
                     full_state=bool(ckpt.get('optimizer_states') and ckpt.get('lr_schedulers')),
                     relation_adapter=any('visible_conditioning.relation_adapter.' in k for k in state))
            found.append(row);print(json.dumps(row),flush=True)
            del ckpt
        except Exception as e:print(json.dumps(dict(path=str(candidate),error=str(e))),flush=True)
    endpoints=[r for r in found if r['epoch']==49 and r['full_state'] and r['relation_adapter']]
    print(json.dumps({'epoch49_candidates':endpoints,'files_checked':len(seen)}),flush=True)

if __name__=='__main__':main()
