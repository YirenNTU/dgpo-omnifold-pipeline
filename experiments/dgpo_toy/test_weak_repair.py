import json
import torch
import pytest
from experiments.dgpo_toy.test_matched_h4 import source
from experiments.dgpo_toy.matched_h4 import load_inputs, run


def test_policy_source_is_explicit_and_retains_weights(source,tmp_path):
    dataset,checkpoint=source
    state=torch.load(checkpoint,weights_only=True)
    policy=tmp_path/"suite/strong_no_kl/state.pt"
    policy.parent.mkdir(parents=True)
    saved={"model":state["model"],"config":state["config"],"step":10000,
           "arm":"strong_no_kl","velocity_coefficient":0.,"source":str(checkpoint)}
    torch.save(saved,policy)
    (policy.parent/"report.json").write_text('{}')
    (policy.parent.parent/"report.json").write_text(json.dumps({"state":"completed",
        "config":state["config"],"arms":{"strong_no_kl":{"steps":10000}},
        "dataset":str(dataset),"source":str(checkpoint)}))
    with pytest.raises(ValueError):load_inputs(dataset,policy)
    _,_,model,_=load_inputs(dataset,policy,policy_source=True)
    assert all(torch.equal(v,state["model"][k]) for k,v in model.state_dict().items())
    report=run(tmp_path/"weak",dataset_path=dataset,diffusion_path=policy,
               feature_mode="plain",policy_source=True,patience=1,min_delta=100.)
    assert report["state"]=="completed" and report["source_unchanged"]
    assert report["source_diffusion_step"]==10000
    saved["velocity_coefficient"]=1.;torch.save(saved,policy)
    with pytest.raises(ValueError,match="pure-strong"):
        load_inputs(dataset,policy,policy_source=True)
