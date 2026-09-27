import torch
from experiments.dgpo_toy.test_matched_h4 import source
from experiments.dgpo_toy.matched_h4 import run as fit
from experiments.dgpo_toy.three_arm_truth import run, ARMS
from experiments.dgpo_toy.conditional import Distribution


def test_three_arms_shared_sources_no_new_truth(source, tmp_path, monkeypatch):
    dataset, diffusion = source
    common = dict(dataset_path=dataset, diffusion_path=diffusion, patience=1, min_delta=100.)
    fit(tmp_path/"strong", **common)
    fit(tmp_path/"weak", **common, feature_mode="plain", panels_from=tmp_path/"strong/panels.pt")
    def forbidden(*args, **kwargs):
        raise AssertionError("No new truth during policy training")
    monkeypatch.setattr(Distribution, "sample", forbidden)
    result = run(tmp_path/"policy", dataset_path=dataset, diffusion_path=diffusion,
        strong_path=tmp_path/"strong/best_classifier.pt", weak_path=tmp_path/"weak/best_classifier.pt", steps=1)
    assert result["state"] == "completed" and result["sources_unchanged"]
    assert set(result["arms"]) == set(ARMS)
    states = {k: torch.load(tmp_path/"policy"/k/"state.pt", weights_only=True) for k in ARMS}
    for k, (_, coefficient) in ARMS.items():
        assert states[k]["step"] == 1 and states[k]["velocity_coefficient"] == coefficient
        assert states[k]["optimizer"]["state"]
    # The velocity penalty and its gradient are zero at the shared initial model.
    for k,v in states["strong_velocity"]["model"].items():
        assert torch.equal(v, states["strong_no_kl"]["model"][k])
