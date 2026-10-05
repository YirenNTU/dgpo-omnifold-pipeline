"""Smooth bounded odds for paired BCE. No clipping of training gradients."""
import math
import numpy as np
from torch.nn import functional as F


def bounded_log_ratio(latent, cap):
    if not math.isfinite(cap) or cap <= 1:
        raise ValueError('Ratio bound must be finite and >1')
    # r=C*sigmoid(z-log(C-1)); z=0 gives r=1. No added parameters/RNG.
    # In unconstrained function space the BCE optimum is min(p/q,C),
    # with the upper boundary approached asymptotically.
    return math.log(cap)-F.softplus(math.log(cap-1)-latent)


def bound_metrics(positive, negative, weight, cap):
    out={}
    for label,s in (('truth',positive),('generated',negative)):
        s=np.asarray(s,float)
        if not np.isfinite(s).all() or (s > np.log(cap)+1e-6).any():
            raise ValueError('Bounded score is nonfinite or above cap')
        fraction=np.exp(s)/cap
        out[label+'_near_cap_fraction']=float(np.average(fraction>=.9,weights=weight))
        out[label+'_mean_output_jacobian']=float(np.average(1-fraction,weights=weight))
        out[label+'_ratio_max']=float(np.exp(s.max()))
    return out
