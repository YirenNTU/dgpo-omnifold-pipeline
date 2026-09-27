"""Exact shared truth rows, fixed negatives, held-out selection and resume."""
from dataclasses import replace

import pytest
import torch

from experiments.dgpo_toy.conditional import Config, Distribution
from experiments.dgpo_toy.truth_pretrain import run as pretrain
from experiments.dgpo_toy.matched_h4 import load_inputs, paired_panels, run
from experiments.dgpo_toy.test_fixed_reward import assert_nested_equal


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    cfg=replace(Config(),dimensions=6,hidden=12,classifier_hidden=12,train_events=16,
        validation_events=8,test_events=10,fit_batch=8,eval_events=8,standardize_step=3)
    dataset=tmp_path/"data.pt"
    pretrain(tmp_path/"diffusion",dataset_path=dataset,cfg=cfg,patience=1,min_delta=100.)
    return dataset,tmp_path/"diffusion/best_model.pt"


def test_exact_shared_data_and_deterministic_fixed_negatives(source):
    cfg,dataset,model,_=load_inputs(*source)
    before=torch.random.get_rng_state().clone()
    a=paired_panels(dataset,model,cfg,17)
    b=paired_panels(dataset,model,cfg,17)
    assert torch.equal(before,torch.random.get_rng_state())
    assert_nested_equal(a,b)
    for name in a:
        assert torch.equal(a[name]["truth"],dataset[name]["target"])
        assert torch.equal(a[name]["context"],dataset[name]["condition"])
        assert a[name]["generated"].shape==a[name]["truth"].shape
    assert all(not p.requires_grad for p in model.parameters())


def test_no_new_truth_no_test_selection_and_frozen_diffusion(source,tmp_path,monkeypatch):
    dataset,checkpoint=source
    before=dataset.read_bytes(),checkpoint.read_bytes()
    monkeypatch.setattr(Distribution,"sample",lambda *a,**k:pytest.fail("Must not generate new truth"))
    import experiments.dgpo_toy.matched_h4 as module
    original=module.classifier_metrics
    counts=[]
    def record(model,panel,data):
        counts.append(len(panel["truth"]))
        return original(model,panel,data)
    monkeypatch.setattr(module,"classifier_metrics",record)
    result=run(tmp_path/"h4",dataset_path=dataset,diffusion_path=checkpoint,patience=1,min_delta=100.)
    assert result["state"]=="completed" and result["completed_epochs"]==2
    assert result["completed_steps"]==8 and result["new_truth_events"]==0
    assert result["source_unchanged"]
    assert counts==[8,8,16,8,10]
    assert (dataset.read_bytes(),checkpoint.read_bytes())==before
    assert all(x["exact_truth_and_context_match"] for x in result["splits"].values())
    assert result["best_validation_bce"]==min(x["validation"]["bce"] for x in result["history"])
    assert result["selected_metrics"]["validation"]["bce"]==result["best_validation_bce"]
    state=torch.load(tmp_path/"h4/last_state.pt",weights_only=True)
    best=torch.load(tmp_path/"h4/best_classifier.pt",weights_only=True)
    assert state["standardized"] and best["feature_mode"]=="joint3"
    assert_nested_equal(state["best_model"],best["classifier"])


def test_resume_exact_with_fixed_panels_and_standardization(source,tmp_path):
    dataset,checkpoint=source
    kw=dict(dataset_path=dataset,diffusion_path=checkpoint,patience=2,min_delta=100.)
    run(tmp_path/"full",**kw)
    run(tmp_path/"partial",stop_after_epoch=1,**kw)
    run(tmp_path/"resume",resume_from=tmp_path/"partial/last_state.pt",**kw)
    a=torch.load(tmp_path/"full/last_state.pt",weights_only=True)
    b=torch.load(tmp_path/"resume/last_state.pt",weights_only=True)
    for key in ("model","optimizer","rng","epoch","step","stale","anchor","best_model","best_loss","standardized"):
        assert_nested_equal(a[key],b[key])
    assert a["report"]["history"]==b["report"]["history"]
    assert a["report"]["selected_metrics"]==b["report"]["selected_metrics"]
    assert_nested_equal(torch.load(tmp_path/"full/panels.pt",weights_only=True),
                        torch.load(tmp_path/"resume/panels.pt",weights_only=True))


@pytest.mark.parametrize("field,value",[("weights","ema"),("step",999),("trained_on","gaussian")])
def test_wrong_diffusion_provenance_rejected(source,field,value):
    dataset,checkpoint=source
    state=torch.load(checkpoint,weights_only=True)
    state[field]=value
    torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="selected raw"):
        load_inputs(dataset,checkpoint)


def test_input_outputs_protected(source):
    dataset,checkpoint=source
    for output in (dataset.parent,checkpoint.parent):
        with pytest.raises(ValueError,match="Separate output"):
            run(output,dataset_path=dataset,diffusion_path=checkpoint)


def test_plain_reuses_identical_strong_panels_without_regeneration(source,tmp_path,monkeypatch):
    dataset,checkpoint=source
    strong=tmp_path/"strong"
    run(strong,dataset_path=dataset,diffusion_path=checkpoint,patience=1,min_delta=100.)
    import experiments.dgpo_toy.matched_h4 as module
    monkeypatch.setattr(module,"paired_panels",lambda *a,**k:pytest.fail("Use saved negatives"))
    monkeypatch.setattr(Distribution,"sample",lambda *a,**k:pytest.fail("No new truth"))
    plain=tmp_path/"plain"
    r=run(plain,dataset_path=dataset,diffusion_path=checkpoint,patience=1,min_delta=100.,
          feature_mode="plain",panels_from=strong/"panels.pt")
    a=torch.load(strong/"panels.pt",weights_only=True)
    b=torch.load(plain/"panels.pt",weights_only=True)
    assert_nested_equal(a,b)
    best=torch.load(plain/"best_classifier.pt",weights_only=True)
    assert best["feature_mode"]==r["feature_mode"]=="plain"
    assert best["classifier"]["encoder.0.weight"].shape[1]==9
