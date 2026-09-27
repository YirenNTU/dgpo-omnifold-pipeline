"""Recover a learned low-ESS Fourier reward with fresh actual-generator data.

Bounded experiment; see LOW_ESS_RECOVERY_PROTOCOL.md. All writes stay in --output.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import itertools
import math
from pathlib import Path
import time

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from .conditional import (Config, Distribution, Denoiser, Classifier, generator,
        initialize, ddim, alpha_sigma, make_panel, scores, classifier_metrics, baseline_health,
        weight_health, policy_panel, policy_train, paired_gain, validity, decide)
except ImportError:
    from conditional import (Config, Distribution, Denoiser, Classifier, generator,
        initialize, ddim, alpha_sigma, make_panel, scores, classifier_metrics, baseline_health,
        weight_health, policy_panel, policy_train, paired_gain, validity, decide)


@torch.no_grad()
def fresh_batch(initial, data, cfg, rng, sample_transform=None):
    """Paired labels at NEW contexts; negatives use actual frozen DDIM weights."""
    c = data.contexts(cfg.fit_batch//2, rng)
    truth = data.sample(c, rng, truth=True)
    noise = torch.randn(len(c), cfg.dimensions, generator=rng)
    generated = ddim(initial, c, noise, cfg.ddim_steps)
    if sample_transform is not None:
        generated = sample_transform(generated, c)
    return (torch.cat([truth, generated]), c.repeat(2, 1),
            torch.cat([torch.ones(len(c)), torch.zeros(len(c))]))


class JointFourierClassifier(Classifier):
    """All signed triple Fourier features, never truth-selected triples.

    Explicit order-three bias: C(d,3) triples, four sign patterns modulo global
    sign, first harmonic only. Original individual features remain available.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg, True)
        triples = torch.tensor(list(itertools.combinations(range(cfg.dimensions), 3)))
        signs = torch.tensor([[1., a, b] for a in [-1., 1.] for b in [-1., 1.]])
        self.register_buffer("triples", triples)
        self.register_buffer("signs", signs)
        size = self.encoder[0].in_features + 2*len(triples)*len(signs)
        self.encoder[0] = torch.nn.Linear(size, cfg.classifier_hidden)

    def features(self, y, c, data):
        base = super().features(y, c, data)
        angles = data.angles(y, c)
        combinations = torch.einsum("...ij,kj->...ik", angles[..., self.triples], self.signs).flatten(-2)
        return torch.cat([base, combinations.sin(), combinations.cos()], -1)


def fit_streaming(initial, data, cfg, standardization_panel, validation, seed, emit,
                  feature_mode="coordinate", sample_transform=None):
    if feature_mode == "coordinate":
        model = initialize(Classifier, cfg, seed+4000, True)
    elif feature_mode == "joint3":
        model = initialize(JointFourierClassifier, cfg, seed+4000)
    else:
        raise ValueError(feature_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.classifier_lr,
                                  weight_decay=cfg.weight_decay)
    rng = generator(seed+81000)
    best, best_step, best_state = math.inf, 0, None
    history = []
    for step in range(1, cfg.classifier_steps+1):
        y, c, labels = fresh_batch(initial, data, cfg, rng, sample_transform)
        if step == cfg.standardize_step:
            count = min(2048, len(standardization_panel["context"]))
            model.standardize(torch.cat([standardization_panel["truth"][:count],
                                         standardization_panel["generated"][:count]]),
                              standardization_panel["context"][:count].repeat(2, 1), data)
            for parameter in model.head.parameters():
                optimizer.state.pop(parameter, None)
        loss = F.binary_cross_entropy_with_logits(model(y, c, data), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        if step % 100 == 0 or step == cfg.classifier_steps:
            values = classifier_metrics(model, validation, data)
            row = {"phase": "streaming_classifier", "step": step,
                   "fresh_train_bce": float(loss.detach()), "gradient_norm": grad,
                   "fresh_contexts_seen": step*(cfg.fit_batch//2), **values}
            history.append(row)
            if values["bce"] < best:
                best, best_step, best_state = values["bce"], step, copy.deepcopy(model.state_dict())
            # All validation measurements remain in report; concise live output.
            if step % 500 == 0 or step == cfg.classifier_steps:
                emit(row)
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, {"selected_step": best_step, "validation_bce": best, "history": history,
                   "fresh_contexts_seen": cfg.classifier_steps*(cfg.fit_batch//2),
                   "feature_mode": feature_mode,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "standardization_source": "original training panel only"}


@torch.no_grad()
def inverse_ddim(model, y: Tensor, c: Tensor, guess: Tensor, steps: int, iterations=20):
    """Fixed-point inversion appropriate to the saved near-identity baseline.

    No assumption of convergence: always return measured endpoint residuals.
    Do not use for an arbitrary evolved/collapsed policy without revalidation.
    """
    z = guess.clone()
    for _ in range(iterations):
        residual = y-ddim(model, c, z, steps)
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Nonfinite DDIM inverse residual")
        z = z+residual
        if float(residual.abs().max()) < 1e-11:
            break
    residual = (y-ddim(model, c, z, steps)).abs().amax(-1)
    return z, residual


def invertibility_certificate(model, steps):
    """Sufficient global inverse-contraction bound for this exact MLP DDIM.

    SiLU has |derivative|<1.1: its positive extremum solves x*tanh(x/2)=2,
    with root<2.4 and derivative .5+x/4<1.1. Each step Ax+Bv(x) is bijective
    if |B|*Lip(v)<A, by the contraction mapping theorem for its inverse.
    Spectral norms are computed in float64; report the margin explicitly.
    """
    if type(model) is not Denoiser:
        return {"certified": False, "reason": "unsupported architecture"}
    layers = list(model.network)
    expected = [torch.nn.Linear, torch.nn.SiLU]*3 + [torch.nn.Linear]
    if [type(layer) for layer in layers] != expected:
        return {"certified": False, "reason": "unsupported layer sequence"}
    d = layers[-1].out_features
    matrices = [layer.weight.detach().double() for layer in layers if isinstance(layer, torch.nn.Linear)]
    matrices[0] = matrices[0][:, :d]
    norms = [float(torch.linalg.matrix_norm(w, ord=2)) for w in matrices]
    bound = math.prod(norms)*1.1**3
    t = torch.arange(steps, 0, -1, dtype=torch.float64)/steps
    a, s = alpha_sigma(t)
    ap, sp = alpha_sigma(t-1/steps)
    aa, bb = ap*a+sp*s, sp*a-ap*s
    margins = aa-bb.abs()*s*bound
    return {"certified": bool((margins > 1e-8).all()), "matrix_spectral_norms": norms,
            "network_lipschitz_upper_bound": bound,
            "minimum_step_margin": float(margins.min()),
            "maximum_inverse_step_contraction": float((bb.abs()*s*bound/aa).max()),
            "arithmetic": "float64 spectral norms, conservative SiLU derivative bound"}


def neural_log_density(model, y, c, mean, steps, chunk=64):
    """Numerical change of variables; independently check global injectivity."""
    if y.dtype != torch.float64:
        raise ValueError("Density diagnostic requires float64")
    certificate = invertibility_certificate(model, steps)
    result, log_dets = [], []
    residual_max, start_disagreement, singular_min = 0., 0., math.inf
    singular_max = 0.
    for yy, cc, mm in zip(y.split(chunk), c.split(chunk), mean.split(chunk)):
        guess = yy-mm
        z, residual = inverse_ddim(model, yy, cc, guess, steps)
        # Opposite displaced starts check more than agreement with a known noise.
        offset = .5 * torch.where(torch.arange(y.shape[-1]) % 2 == 0, 1., -1.).to(y)
        for sign in [-1, 1]:
            other, other_residual = inverse_ddim(model, yy, cc, guess+sign*offset, steps)
            start_disagreement = max(start_disagreement, float((other-z).abs().max()))
            residual_max = max(residual_max, float(other_residual.max()))
        jacobian = torch.vmap(torch.func.jacrev(lambda zz, context: ddim(model, context, zz, steps)))(z, cc)
        sign, logdet = torch.linalg.slogdet(jacobian)
        singular = torch.linalg.svdvals(jacobian)
        if (sign == 0).any() or not torch.isfinite(logdet).all():
            raise FloatingPointError("Singular/nonfinite neural DDIM Jacobian")
        singular_min = min(singular_min, float(singular.min()))
        singular_max = max(singular_max, float(singular.max()))
        residual_max = max(residual_max, float(residual.max()))
        result.append(-.5*(z.square()+math.log(2*math.pi)).sum(-1)-logdet)
        log_dets.append(logdet)
    return torch.cat(result).detach(), {
        "inverse_max_residual": residual_max,
        "inverse_start_disagreement": start_disagreement,
        "min_singular_value": singular_min, "max_singular_value": singular_max,
        "mean_log_abs_det": float(torch.cat(log_dets).mean()),
        "global_injectivity_certified": certificate["certified"],
        "invertibility_certificate": certificate,
    }


def density_diagnostic(initial, critic, data, cfg, n, seed):
    panel = make_panel(initial, data, cfg, n, seed)
    model = copy.deepcopy(initial).double().eval().requires_grad_(False)
    log_reference, learned, per_class = [], [], {}
    for name in ["truth", "generated"]:
        y, c = panel[name].double(), panel["context"].double()
        mean = data.mean(c)
        log_neural, diagnostics = neural_log_density(model, y, c, mean, cfg.ddim_steps)
        log_nominal = -.5*((y-mean).square()+math.log(2*math.pi)).sum(-1)
        delta = log_neural-log_nominal
        # p(y|c) is known exactly (up to the sampler's negligible CDF clamp).
        reference = data.nominal_log_ratio(y, c)-delta
        with torch.no_grad():
            fitted = critic(panel[name], panel["context"], data).double()
        error = fitted-reference
        diagnostics.update({"log_density_delta_rms": float(delta.square().mean().sqrt()),
                            "log_density_delta_max_abs": float(delta.abs().max()),
                            "logit_error_rms": float(error.square().mean().sqrt()),
                            "logit_error_mean": float(error.mean()),
                            "logit_error_q95_abs": float(error.abs().quantile(.95)),
                            "learned_weight_health": weight_health(fitted),
                            "numerical_reference_weight_health": weight_health(reference)})
        per_class[name] = diagnostics
        log_reference.append(reference)
        learned.append(fitted)
    bce = lambda v: float(.5*(F.softplus(-v[0]).mean()+F.softplus(v[1]).mean()))
    certified = all(v["global_injectivity_certified"] for v in per_class.values())
    return {"contexts_per_class": n,
            "density_kind": "numerical_certified_bijection" if certified else "numerical_single_branch",
            "global_injectivity_certified": certified, "classes": per_class,
            "reference_bce": bce(log_reference), "learned_bce": bce(learned),
            "bce_excess": bce(learned)-bce(log_reference)}


def density_gates(diagnostic):
    values = list(diagnostic["classes"].values())
    return {"inverse_converged": all(v["inverse_max_residual"] <= 1e-8 for v in values),
            "inverse_start_agreement": all(v["inverse_start_disagreement"] <= 1e-7 for v in values),
            "jacobian_condition": all(v["min_singular_value"] >= .25 for v in values),
            "reference_bce_agreement": diagnostic["bce_excess"] <= .10,
            "truth_logit_rms": diagnostic["classes"]["truth"]["logit_error_rms"] <= 1.}


@torch.no_grad()
def confirmation_metrics(critic, panel, data):
    metrics = classifier_metrics(critic, panel, data)
    positive, negative = scores(critic, panel, data)
    n = len(negative)
    metrics["mean_ratio_relative_se"] = math.sqrt(max(0., (1/metrics["ess_fraction"]-1)/(n-1)))
    metrics["chunks"] = [weight_health(part) for part in negative.tensor_split(8)]
    metrics["contexts_per_class"] = n
    return metrics


def run(source, output, seed=17, classifier_steps=20000, smoke=False, feature_mode="coordinate"):
    started = time.perf_counter()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    cfg = replace(Config(**payload["config"]), classifier_steps=classifier_steps)
    if smoke:
        cfg = replace(cfg, classifier_steps=3, fit_batch=16, train_events=128,
                      validation_events=64, eval_events=16, batch=4,
                      timesteps=2, policy_steps=2, eval_every=2, standardize_step=2)
    initial = initialize(Denoiser, cfg, seed)
    initial.load_state_dict(payload["initial"])
    initial.eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    log = (output/"progress.jsonl").open("w")
    def emit(row):
        text = json.dumps({"seed": seed, **row}, allow_nan=False)
        print(text, flush=True)
        log.write(text+"\n")
        log.flush()
    report = {"source": str(source.resolve()), "seed": seed, "smoke": smoke,
              "config": asdict(cfg), "sampling_intervention": "fresh_actual_generator",
              "feature_mode": feature_mode,
              "endpoints": {}, "policy_histories": {}, "decision": "running"}
    def save():
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    try:
        data = Distribution(cfg)
        standardization = make_panel(initial, data, cfg, cfg.train_events, seed+10000)
        validation = make_panel(initial, data, cfg, cfg.validation_events, seed+20000)
        critic, fit = fit_streaming(initial, data, cfg, standardization, validation, seed, emit, feature_mode)
        report["fit"] = fit
        torch.save({"config": asdict(cfg), "source": str(source.resolve()),
                    "initial": initial.state_dict(), "classifier": critic.state_dict(),
                    "selected_step": fit["selected_step"], "feature_mode": feature_mode}, output/"reward.pt")
        save()
        n_test, n_density = (128, 8) if smoke else (131072, 2048)
        test = make_panel(initial, data, cfg, n_test, seed+73000)
        report["confirmation"] = confirmation_metrics(critic, test, data)
        report["baseline"] = baseline_health(test["generated"], test["context"], data)
        report["joint_gap"] = float(data.joint_signal(test["truth"], test["context"]).mean()
                                     - data.joint_signal(test["generated"], test["context"]).mean())
        emit({"phase": "confirmation", "selected_step": fit["selected_step"],
              **{k: v for k, v in report["confirmation"].items() if k != "chunks"}})
        save()
        emit({"phase": "density_diagnostic_start", "contexts_per_class": n_density})
        report["density"] = density_diagnostic(initial, critic, data, cfg, n_density, seed+74000)
        rewards, _ = policy_panel(initial, critic, data, cfg, seed+75000)
        informative = float(((rewards.max(1).values-rewards.min(1).values)>1e-3).float().mean())
        gates = validity(report["baseline"], report["confirmation"], informative, report["joint_gap"])
        gates.update(density_gates(report["density"]))
        report["validity"] = gates
        report["nominal_gaussian_population_ess"] = data.nominal_ess()
        emit({"phase": "setup_validity", "gates": gates, "density": report["density"]})
        save()
        if all(gates.values()) or smoke:
            endpoint_initial, _ = policy_panel(initial, critic, data, cfg, seed+60000)
            for arm in ["dgpo", "pathwise"]:
                model, history = policy_train(arm, initial, critic, data, cfg, seed, seed+50000, emit)
                endpoint, _ = policy_panel(model, critic, data, cfg, seed+60000)
                report["endpoints"][arm] = paired_gain(endpoint, endpoint_initial)
                report["policy_histories"][arm] = history
                torch.save(model.state_dict(), output/(arm+".pt"))
                save()
        # This is a read-only source checkpoint throughout classifier training.
        if any(not torch.equal(value, payload["initial"][key]) for key, value in initial.state_dict().items()):
            raise RuntimeError("Frozen initial generator changed")
        report["decision"] = decide(gates, report["endpoints"], smoke)
        report["seconds"] = time.perf_counter()-started
        save()
        emit({"phase": "complete", "decision": report["decision"],
              "endpoints": report["endpoints"], "seconds": report["seconds"]})
        return report
    except BaseException as exc:
        report["decision"] = "interrupted_or_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Previous conditional toy models.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--classifier-steps", type=int, default=20000)
    parser.add_argument("--feature-mode", choices=["coordinate", "joint3"], default="coordinate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.classifier_steps < 1:
        parser.error("--classifier-steps must be positive")
    if args.output.resolve() == args.source.resolve().parent:
        parser.error("Use a separate output to preserve the source experiment")
    torch.set_num_threads(1)
    run(args.source, args.output, args.seed, args.classifier_steps, args.smoke, args.feature_mode)


if __name__ == "__main__":
    main()
