"""Function-preserving nonlinear FiLM generators for the General-only ablation."""
import torch
from torch import nn
from torch.nn import functional as F


def film_mlp_spec():
    return dict(version=1, context_width=256, hidden_width=128, output_width=128,
                blocks=3, activation='silu', preserve_linear=True,
                zero_output=True, cross_attention=False, general_head_trainable=True,
                backbone_trainable=False)


class NonlinearFiLMProjection(nn.Linear):
    """Keep original weight key/function; learn an additive nonlinear correction."""
    def __init__(self, original):
        if original.in_features != 256 or original.out_features != 128 or original.bias is not None:
            raise ValueError('Expected the original 256-to-128 bias-free FiLM projection')
        super().__init__(256,128,bias=False)
        self.weight = original.weight
        self.mlp = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,128))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, context):
        return F.linear(context,self.weight)+self.mlp(context)


def add_film_mlp(model):
    if len(model.blocks) != 3 or hasattr(model,'cross_attention'):
        raise ValueError('Requires the original three-block General without cross-attention')
    for block in model.blocks:
        if block.kind != 'film' or type(block.context) is not nn.Linear:
            raise ValueError('Requires unmodified FiLM blocks')
        block.context = NonlinearFiLMProjection(block.context)
    return model
