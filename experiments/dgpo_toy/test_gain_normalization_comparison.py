from experiments.dgpo_toy.compare_gain_normalization import within_margins


def test_nonsignificance_is_not_equivalence():
    margins={'auc_gap':.005,'bce':.003}
    narrow={'auc_gap':{'lo95':-.004,'hi95':.004},'bce':{'lo95':-.002,'hi95':.002}}
    assert within_margins(narrow,margins)
    wide={**narrow,'auc_gap':{'lo95':-.01,'hi95':.01}}
    assert not within_margins(wide,margins)
    one_fails={**narrow,'bce':{'lo95':-.001,'hi95':.0031}}
    assert not within_margins(one_fails,margins)
