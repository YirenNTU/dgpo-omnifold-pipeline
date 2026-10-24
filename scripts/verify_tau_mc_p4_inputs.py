"""Read-only CPU audit of the actual MC p4/offset join in filtered parquet."""
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'evenet_dgpo')]
from RL.DGPO_neutrino.tau_oracle_mixture import MC_TAU_COLUMNS, validate_mc_tau_p4, directions


def verify_inputs(inputs):
    reports=[]
    columns=list(MC_TAU_COLUMNS)
    columns += [f'lead_{leg}_visible_{q}' for leg in 'ab' for q in ('px','py','pz')]
    columns += [f'x_invisible:{leg}:{q}' for leg in (0,1) for q in (0,1)]
    for directory,expected_rows in inputs:
        rows=0;worst=0.;energy_min=float('inf');energy_max=0.
        files=sorted(Path(directory).glob('*.parquet'))
        if not files:raise ValueError('No filtered parquet files: '+str(directory))
        for path in files:
            parquet=pq.ParquetFile(path)
            missing=set(columns)-set(parquet.schema_arrow.names)
            if missing:raise ValueError(f'{path}: missing required MC/geometry fields: {sorted(missing)}')
            for rb in parquet.iter_batches(batch_size=8192,columns=columns):
                a={name:np.asarray(rb.column(i).to_numpy(zero_copy_only=False),dtype=np.float64)
                   for i,name in enumerate(rb.schema.names)}
                p4=torch.from_numpy(np.stack([a[k] for k in MC_TAU_COLUMNS],-1).reshape(-1,2,4))
                validate_mc_tau_p4(p4)
                visible=torch.from_numpy(np.stack([a[k] for k in columns[8:14]],-1).reshape(-1,2,3))
                offsets=torch.from_numpy(np.stack([a[k] for k in columns[14:]],-1).reshape(-1,2,2))
                if (not torch.isfinite(visible).all() or not torch.isfinite(offsets).all()
                        or (torch.linalg.vector_norm(visible,dim=-1)<=0).any()):
                    raise ValueError(f'{path}: invalid physical visible/target inputs')
                unit=p4[...,1:]/torch.linalg.vector_norm(p4[...,1:],dim=-1,keepdim=True)
                error=float(torch.linalg.vector_norm(directions(offsets,visible)-unit,dim=-1).max())
                if error>1e-4:raise ValueError(f'{path}: MC p4/ordered target direction mismatch {error}')
                worst=max(worst,error);rows+=len(rb)
                energy_min=min(energy_min,float(p4[...,0].min()))
                energy_max=max(energy_max,float(p4[...,0].max()))
        if rows!=expected_rows:raise ValueError(f'{directory}: {rows} != {expected_rows} events')
        report=dict(directory=str(directory),rows=rows,files=len(files),mc_energy_GeV=[energy_min,energy_max],
                    max_target_direction_chord_error=worst,actual_mc_p4=True)
        reports.append(report)
        print('MC P4 INPUTS OK: '+json.dumps(report),flush=True)
    return reports


if __name__=='__main__':
    from scripts.train_diffusion_film_10pct import TRAIN,VAL
    verify_inputs([(TRAIN,416701),(VAL,119002)])
