"""Integrity checks for the step500 fixed General/reference intervention."""
import torch


def assert_same_state(actual, expected, name):
    if set(actual) != set(expected) or any(
        not torch.equal(actual[k].detach().cpu(), expected[k].detach().cpu())
        for k in expected
    ):
        raise ValueError(f'Fixed General round: {name} changed')


def validate_fixed_checkpoint(payload, anchor):
    """Allow interrupted continuation, but never a teacher/reference refit."""
    if not 500 <= payload['global_step'] <= 800:
        raise ValueError('Fixed General round requires a checkpoint between steps500 and800')
    for name in ('round_id', 'denominator_step', 'last_refit_epoch'):
        if payload['dgpo_omnifold_reward_stack'][name] != anchor['dgpo_omnifold_reward_stack'][name]:
            raise ValueError(f'Fixed General round: saved {name} changed')
    assert_same_state(payload['dgpo_omnifold_reward_stack']['head']['state_dict'],
                      anchor['dgpo_omnifold_reward_stack']['head']['state_dict'], 'saved General')
    assert_same_state(payload['dgpo_ref_state_dict'], anchor['dgpo_ref_state_dict'], 'saved reference')


class FixedGeneralRound:
    def __init__(self, reward, reference):
        self.reward, self.reference = reward, reference
        self.head_state = {k: v.detach().cpu().clone() for k, v in reward.head.state_dict().items()}
        self.ref_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
        self.verify()

    def verify(self):
        r = self.reward
        if (r.round_id, r.denominator_step, r.last_refit_epoch) != (6, 500, 49):
            raise ValueError('Fixed General round requires original step500 General round6')
        if any(p.requires_grad for model in (r.head, self.reference) for p in model.parameters()):
            raise ValueError('Fixed General and reference must have frozen parameters')
        assert_same_state(r.head.state_dict(), self.head_state, 'General')
        assert_same_state(self.reference.state_dict(), self.ref_state, 'reference')
