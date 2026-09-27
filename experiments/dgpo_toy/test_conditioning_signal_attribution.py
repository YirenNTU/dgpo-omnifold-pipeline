import torch

from experiments.dgpo_toy.conditioning_signal_attribution import compare_arrays


def test_crossed_contrasts_keep_same_judge_and_cancel_truth_anchor():
    arrays={}
    for pi,arm in enumerate(('baseline','raw','fourier')):
        arrays[arm]={'structure':{f'order{k}_mode_l2_debiased':torch.full((8,),pi*.01*k,dtype=torch.float64) for k in (1,2,3)},'judges':{}}
        for ji,judge in enumerate(('baseline','raw','fourier')):
            arrays[arm]['judges'][judge]={'bce':{s:torch.full((8,),ji+pi*(0 if s=='A' else .1),dtype=torch.float64) for s in 'ABCD'}}
    r=compare_arrays(arrays,100)
    for key,v in r['bce'].items():
        assert abs(v['mean']-.1)<1e-12,key
        assert abs(v['hi95']-v['lo95'])<1e-12
    assert abs(r['structure']['order3_mode_l2_debiased/fourier_minus_raw']['mean']-.03)<1e-12
