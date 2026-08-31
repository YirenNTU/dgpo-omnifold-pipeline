"""Ztautau adaptation of the current EveNet conditional OmniFold fitter."""

from .evenet_ratio import (
    EventPackingSpec,
    EvenetAdapterModelBuilder,
    FrozenResidualRatioReward,
    fit_residual_ratio_stack,
    pack_event_inputs,
    peft_bank_factory,
)
from .ratio_fit import RatioFitConfig
from .dgpo_reward import ZtautauOmniFoldReward, load_ztautau_omnifold_reward
from .adaptive import (
    AdaptiveOmniFoldConfig,
    AdaptiveOmniFoldPool,
    AdaptiveOmniFoldState,
)

__all__ = [
    "EventPackingSpec",
    "EvenetAdapterModelBuilder",
    "FrozenResidualRatioReward",
    "RatioFitConfig",
    "ZtautauOmniFoldReward",
    "AdaptiveOmniFoldConfig",
    "AdaptiveOmniFoldPool",
    "AdaptiveOmniFoldState",
    "fit_residual_ratio_stack",
    "pack_event_inputs",
    "peft_bank_factory",
    "load_ztautau_omnifold_reward",
]
