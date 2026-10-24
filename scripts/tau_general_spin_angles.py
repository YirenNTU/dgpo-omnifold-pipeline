"""Single jointly fitted General + calibrated Spin + raw Tau-angle ratio."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.tau_tihsu_cij import AXES, PROTOCOL, from_deltas
from scripts.tau_topology_branch import topology_features, FIELDS
from scripts.tau_exporter_cij import PROTOCOL as ROOT_PROTOCOL
from scripts.tau_root_export_geometry import panel_protocol, target_spin_observables, prediction_spin_observables


def branch_spec(base_dim=533, protocol=PROTOCOL):
    if protocol not in (PROTOCOL, ROOT_PROTOCOL):
        raise ValueError('Unknown Spin reconstruction protocol')
    spec = dict(version=1, base_candidate_dim=base_dim, width=64,
                spin_protocol=protocol, spin_dim=15, tau_fields=FIELDS,
                tau_calibration=False, spin_calibration=True, zero_output=True,
                fit_general=True)
    if protocol == ROOT_PROTOCOL:
        spec.update(version=2, spin_calibration='prediction_only',
                    target_map='truth_tau_p4_plus_reco_visible',
                    prediction_map='root_evenet_common_calibration',
                    angular_domain='cartesian_periodic_extension')
    return spec


def _features(o, va, vb, delta):
    names = [f'cos_theta_{leg}_{axis}' for leg in ('A', 'B') for axis in AXES]
    names += [f'cos_theta_A_{a}_times_cos_theta_B_{b}' for a in AXES for b in AXES]
    spin = np.stack([o[name] for name in names], -1)
    result = np.c_[spin, topology_features(va, vb, delta)].astype(np.float32)
    if result.shape != (len(delta), 29) or not np.isfinite(result).all():
        raise ValueError('Invalid candidate-owned Spin/Tau features')
    return result


def features(va, vb, delta, *, protocol=PROTOCOL):
    """Generated features: no truth arguments or access to panel truth p4."""
    branch_spec(protocol=protocol)
    if protocol == ROOT_PROTOCOL:
        from scripts.tau_root_export_geometry import prediction_observables
        o = prediction_observables(va, vb, delta)
    else:
        o = from_deltas(va, vb, delta)
    return _features(o, va, vb, delta)


def truth_features(panel, indices=slice(None)):
    return _features(target_spin_observables(panel, indices),
        panel['visible_a'][indices], panel['visible_b'][indices], panel['truth_deltas'][indices])


def validate_truth_cache(panel, spec):
    """Same dimensions are insufficient: old and corrected maps both have 29 inputs."""
    if panel_protocol(panel) != spec['spin_protocol']:
        raise ValueError('Truth cache Spin protocol differs from classifier')
    if spec['spin_protocol'] == ROOT_PROTOCOL:
        n = len(panel['source_ids'])
        if any(k not in panel or panel[k].shape != (n, 4)
               or not np.isfinite(panel[k]).all() for k in ('truth_tau_a', 'truth_tau_b')):
            raise ValueError('Root target truth p4 cache is missing or invalid')
        if bool(panel.get('truth_features_ready', False)):
            width = spec['base_candidate_dim']
            if panel['candidate_truth'].shape != (n, width+29):
                raise ValueError('Root target feature cache shape differs')
            for start in range(0, n, 4096):
                sl = slice(start, start+4096)
                np.testing.assert_allclose(panel['candidate_truth'][sl, width:], truth_features(panel, sl),
                                           atol=1e-6, rtol=1e-6)


class Branch(nn.Module):
    def __init__(self, dimension, condition_dim, classes):
        super().__init__()
        from scripts.tau_conditioning_heads import ConditioningBlock
        self.condition = nn.Sequential(nn.Linear(condition_dim, 64), nn.SiLU())
        self.label = nn.Linear(classes, 64, bias=False)
        self.input = nn.Sequential(nn.Linear(dimension, 64), nn.SiLU())
        self.block = ConditioningBlock('film', 64, 0., 64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, c, labels):
        h = self.input(x)
        context = self.condition(c) + self.label(labels)
        return self.output(h + self.block(h, context)).flatten()


class GeneralSpinAngles(nn.Module):
    def __init__(self, base, spec, cap):
        super().__init__()
        if spec != branch_spec(spec['base_candidate_dim'], spec['spin_protocol']) or cap != 30 or base.process_label_index is None:
            raise ValueError('Requires declared calibrated Spin/raw Tau, process labels, cap30')
        self.base = base
        self.base_dim = spec['base_candidate_dim']
        self.ratio_bound = cap
        self.classes = base.num_process_classes
        with torch.random.fork_rng(devices=[]):
            self.spin = Branch(15, base.condition_encoder[0].in_features, self.classes)
            self.tau = Branch(14, base.condition_encoder[0].in_features, self.classes)

    def forward(self, condition, candidate):
        if candidate.shape[1] != self.base_dim + 29:
            raise ValueError('Wrong General/Spin/Tau feature layout')
        score = self.base(condition, candidate[:, :self.base_dim])
        c = condition[:, self.base.continuous_condition_columns]
        labels = F.one_hot(condition[:, self.base.process_label_index].long(), self.classes).to(c.dtype)
        score = score + self.spin(candidate[:, self.base_dim:self.base_dim+15], c, labels)
        score = score + self.tau(candidate[:, self.base_dim+15:], c, labels)
        from scripts.tau_bounded_ratio import bounded_log_ratio
        return bounded_log_ratio(score, self.ratio_bound)


def head_config(base, protocol=PROTOCOL):
    cfg = copy.deepcopy(base)
    if any(cfg.get(k) for k in ('physics_branch', 'topology_branch', 'explicit_input', 'cross_attention')):
        raise ValueError('Requires the original General schema')
    cfg.update(candidate_dim=base['candidate_dim']+29,
               general_spin_angles=branch_spec(base['candidate_dim'], protocol))
    return cfg


def build(cfg, condition_dim=None, candidate_dim=None):
    from scripts.train_conditional_spin_ratio import build_classifier
    spec = cfg['general_spin_angles']
    dim = cfg['candidate_dim'] if candidate_dim is None else candidate_dim
    if dim != spec['base_candidate_dim']+29:
        raise ValueError('Wrong combined candidate dimension')
    base_cfg = copy.deepcopy(cfg)
    base_cfg.update(general_spin_angles=None, candidate_dim=spec['base_candidate_dim'], ratio_bound=None)
    return GeneralSpinAngles(build_classifier(base_cfg, condition_dim=condition_dim), spec, cfg['ratio_bound'])
