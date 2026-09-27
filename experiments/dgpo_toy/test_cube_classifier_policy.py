import torch
from experiments.dgpo_toy.conditional import Config, Denoiser, fit_classifier, make_panel
from experiments.dgpo_toy.parity_cube import CubeDistribution
from experiments.dgpo_toy.cube_classifier_policy import CubeFourierClassifier, ordering_diagnostics, evaluate


def config():
    return Config(dimensions=3, context_dim=1, hidden=8, classifier_hidden=8,
        classifier_steps=2, fit_batch=16, standardize_step=1,
        ddim_steps=2, eval_events=16, candidates=4)


def test_fourier_features_have_no_oracle_dependency():
    cfg = config(); model = CubeFourierClassifier(cfg, True)
    y = torch.randn(2, 4, 3); c = torch.randn(2, 4, 1)
    assert model(y, c, None).shape == (2, 4)
    assert torch.equal(model.features(y, c, None)[..., 1:4], y)


def test_strict_ordering_ignores_oracle_ties():
    oracle = torch.tensor([[1., 1., -1., -1.]])
    assert ordering_diagnostics(oracle, oracle)["strict_pair_agreement"] == 1.
    assert ordering_diagnostics(-oracle, oracle)["strict_pair_agreement"] == 0.
    assert ordering_diagnostics(oracle, torch.ones_like(oracle))["centered_cosine"] is None


def test_fit_factory_and_paired_evaluation():
    torch.set_num_threads(1)
    cfg = config(); data = CubeDistribution(cfg, continuous=True, reference_sharpness=4.)
    model = Denoiser(cfg)
    train = make_panel(model, data, cfg, 32, 1)
    val = make_panel(model, data, cfg, 32, 2)
    critic, fit = fit_classifier(cfg, data, train, val, 3, True, lambda row: None,
        model_class=CubeFourierClassifier)
    assert fit["selected_step"] == 2
    assert not any(p.requires_grad for p in critic.parameters())
    first = evaluate(model, critic, data, cfg, 4)
    second = evaluate(model, critic, data, cfg, 4)
    assert all(torch.equal(a, b) for a, b in zip(first[:3], second[:3]))
    assert len(first[3]["conditions"]) == 8
