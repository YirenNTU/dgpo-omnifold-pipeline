import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

from typing import Optional, List

from evenet.network.body.embedding import FourierEmbedding, PointCloudPositionalEmbedding
from evenet.network.layers.activation import create_activation
from evenet.network.layers.linear_block import ResNetDense
from evenet.network.layers.utils import StochasticDepth
from evenet.network.layers.transformer import GeneratorTransformerBlockModule
from evenet.network.layers.activation import create_residual_connection
from evenet.network.body.visible_conditioning import VisibleConditioning


class EventGenerationHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            projection_dim: int,
            num_global_cond: int,
            num_classes: int,
            output_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
            layer_scale: bool,
            layer_scale_init: float,
            drop_probability: float,
            feature_drop: float,
            position_encode: bool = False,
            max_position_length: int = 20,
            visible_conditioning: Optional[dict] = None,
            identity_init_from_layer: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes

        self.time_embedding = FourierEmbedding(projection_dim)
        self.num_point_cloud_embedding = FourierEmbedding(projection_dim)

        self.bridge_point_cloud = create_residual_connection(
            skip_connection=True,
            input_dim=input_dim,
            output_dim=projection_dim
        )

        self.position_encode = position_encode
        if self.position_encode:
            self.position_encoder = PointCloudPositionalEmbedding(
                num_points=max_position_length,
                embed_dim=projection_dim
            )

        self.bridge_global_cond = create_residual_connection(
            skip_connection=True,
            input_dim=num_global_cond,
            output_dim=projection_dim,
        )

        self.cond_token = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(approximate='none')
        )
        # self.label_embedding = nn.Embedding(num_classes, projection_dim)
        self.label_dense = nn.Linear(num_classes, projection_dim, bias=False)
        self.feature_drop = feature_drop
        self.stochastic_depth = StochasticDepth(feature_drop)
        self.gen_transformer_blocks = nn.ModuleList([
            GeneratorTransformerBlockModule(
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_scale=layer_scale,
                layer_scale_init=layer_scale_init,
                drop_probability=drop_probability)
            for _ in range(num_layers)
        ])
        self.generator = nn.Linear(projection_dim, output_dim)
        self.visible_conditioning = VisibleConditioning(**visible_conditioning) if visible_conditioning else None
        if (self.visible_conditioning is not None
                and self.visible_conditioning.pair_attention is not None
                and self.visible_conditioning.pair_attention.heads != num_heads):
            raise ValueError("pair bias heads must match generation attention heads")
        if identity_init_from_layer is not None:
            if (isinstance(identity_init_from_layer, bool)
                    or not isinstance(identity_init_from_layer, int)
                    or not 0 <= identity_init_from_layer < num_layers):
                raise ValueError("identity_init_from_layer must be a zero-based block index below num_layers")
            for block in self.gen_transformer_blocks[identity_init_from_layer:]:
                block.initialize_as_identity()

    def forward(self,
                x,
                global_cond,
                global_cond_mask,
                num_x,
                x_mask,
                time,
                label,
                attn_mask=None,
                time_masking=None,
                position_encode=False,
                visible_raw=None,
                visible_tokens=None,
                visible_mask=None,
                visible_normalized=None):
        """
        x: [B, T, D] <- Noised Input
        global_cond: [B, 1, D] <- Global Condition
        global_cond_mask: [B, 1, 1] <- Mask
        num_x: [B, 1] <- Number of points_cloud
        x_mask: [B, T, 1] <- Mask
        time: [B,] <- Time
        label: [B, 1] <- Conditional Label, one-hot in function,
        time_masking: [B, T, 1] <- Mask for time embedding
        """
        time_emb = self.time_embedding(time).unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, 1, proj_dim]
        if time_masking is not None:
            time_emb = time_emb * time_masking

        cond_token = self.cond_token(
            torch.cat(
                [time_emb, (self.bridge_global_cond(global_cond) * global_cond_mask).expand(-1, x.shape[1], -1)],
                dim=-1
            ),
        )  # After MLP, cond_token shape: torch.Size([B, 1, proj_dim])
        x = self.bridge_point_cloud(x) * x_mask

        if position_encode:
            x = self.position_encoder(
                x=x,
                x_mask=x_mask,
                time_mask=time_masking
            )

        if num_x is not None:
            num_x_embed = self.num_point_cloud_embedding(num_x).unsqueeze(1)  # [B, 1, proj_dim]
            cond_token = cond_token + num_x_embed
            # TODO: Check if this works

        if label is not None:
            label = F.one_hot(label, num_classes=self.num_classes).to(
                dtype=self.label_dense.weight.dtype)  # [B, 1, C]
            label_emb = self.label_dense(label)  # [B, 1, D]
            cond_token = cond_token + label_emb

        else:
            print("ERROR: In Generation Head, Label is None, skipping label embedding")

        modulations = None
        token_memory = None
        pair_biases = None
        if self.visible_conditioning is not None:
            if any(value is None for value in (visible_raw, visible_tokens, visible_mask, time_masking)):
                raise ValueError("TruthGeneration visible conditioning requires observed context and invisible mask")
            extra = (dict(normalized_raw=visible_normalized)
                     if self.visible_conditioning.feature_mode == "kinematics" else {})
            if self.visible_conditioning.token_readout is not None:
                modulations, token_memory, token_mask = self.visible_conditioning(
                    visible_raw, visible_tokens, visible_mask, return_tokens=True, **extra)
            elif self.visible_conditioning.pair_attention is not None:
                modulations, pair_biases = self.visible_conditioning(
                    visible_raw, visible_tokens, visible_mask, return_attention_bias=True, **extra)
            else:
                modulations = self.visible_conditioning(visible_raw, visible_tokens, visible_mask, **extra)
        for index, transformer_block in enumerate(self.gen_transformer_blocks):
            concatenated = cond_token + x
            modulation = modulations[index] if modulations is not None else None
            direct_residual = None
            readout = self.visible_conditioning.token_readout if token_memory is not None else None
            if (readout is not None and readout.mode != "velocity"
                    and (readout.block == "all" or index == len(self.gen_transformer_blocks) - 1)):
                # Visible-first layout is the existing EveNet generation contract.
                # Queries contain noisy state, time and slot information, never clean truth.
                n_visible = token_memory.shape[1]
                residual = readout(
                    concatenated[:, n_visible:], token_memory,
                    (time_masking.bool() & x_mask.bool())[:, n_visible:], token_mask)
                if readout.mode == "film":
                    modulation = tuple(global_part[:, None, :] + F.pad(local, (0, 0, n_visible, 0))
                                       for global_part, local in zip(modulation, residual))
                else:
                    direct_residual = F.pad(residual, (0, 0, n_visible, 0))
                if self.visible_conditioning.log_diagnostics:
                    if readout.mode == "film":
                        self.visible_conditioning.diagnostics.update(readout.diagnostics)
                    else:
                        self.visible_conditioning.diagnostics.update(
                            {f"block_{index}/{key}": value for key, value in readout.diagnostics.items()})
            if modulation is not None:
                # Also handles future combinations with token-specific FiLM.
                modulation = self.visible_conditioning.apply_scale_policy(modulation)
                # Apply numeric gains once; applying them in the encoder too
                # would square nonbinary values.
                modulation = self.visible_conditioning.apply_runtime_gains(modulation)
            pair_kwargs = (dict(attention_bias=pair_biases[index])
                           if pair_biases is not None and index < len(pair_biases) else {})
            out_x, cond_token = transformer_block(
                concatenated, cond_token, x_mask, attn_mask=attn_mask,
                modulation=modulation,
                modulation_mask=(time_masking.bool() & x_mask.bool()) if modulations is not None else None,
                **pair_kwargs,
            )
            if direct_residual is not None:
                # The legacy generator propagates cond_token, not out_x, between
                # blocks. Write here after the existing FFN/LayerScale, without
                # a second zero gate. Only valid invisible slots receive updates.
                cond_token = cond_token + direct_residual
        final_hidden = cond_token + x
        x = F.layer_norm(final_hidden, [final_hidden.size(-1)])
        x = self.generator(x)
        if token_memory is not None and self.visible_conditioning.token_readout.mode == "velocity":
            readout = self.visible_conditioning.token_readout
            n_visible = token_memory.shape[1]
            valid = (time_masking.bool() & x_mask.bool())[:, n_visible:]
            correction = readout(
                final_hidden[:, n_visible:], token_memory,
                valid, token_mask)
            correction = F.pad(correction, (0, 0, n_visible, 0))
            if correction.shape != x.shape:
                raise ValueError("velocity conditioning output does not match generated velocity shape")
            if self.visible_conditioning.log_diagnostics:
                mask = (time_masking.bool() & x_mask.bool()).expand_as(x)
                denom = mask.sum().clamp_min(1)
                base_rms = (torch.where(mask, x.detach().float(), 0.).square().sum() / denom).sqrt()
                diagnostics = dict(readout.diagnostics)
                diagnostics["velocity_residual_to_base_rms"] = (
                    diagnostics["velocity_residual_rms"] / base_rms.clamp_min(1e-8))
                self.visible_conditioning.diagnostics.update(diagnostics)
            x = x + correction

        return x * x_mask


class GlobalCondGenerationHead(nn.Module):

    def __init__(
            self,
            num_layer: int,
            num_resnet_layer: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            input_cond_indices: List[int],
            num_classes: int,
            resnet_dim: int,
            layer_scale_init: float,
            feature_drop_for_stochastic_depth: float,
            activation: str,
            dropout: float
    ):
        super(GlobalCondGenerationHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = resnet_dim
        self.layer_scale_init = layer_scale_init
        self.dropout = dropout
        self.resnet_num_layer = num_resnet_layer
        self.activation = activation
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.input_cond_indices = input_cond_indices
        self.input_cond_dim = len(input_cond_indices)

        self.fourier_projection = FourierEmbedding(self.hidden_dim)
        self.dense_t = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )

        if self.input_cond_dim > 0:
            self.global_cond_embedding = nn.Sequential(
                nn.Linear(self.input_cond_dim, self.hidden_dim),
                create_activation(self.activation, self.hidden_dim)
            )
        if num_classes > 0:
            self.label_embedding = nn.Sequential(
                nn.Linear(num_classes, 2 * self.hidden_dim, bias=False),
                StochasticDepth(feature_drop_for_stochastic_depth)
            )

        self.cond_token_embedding = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            create_activation(self.activation, self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            create_activation(self.activation, self.hidden_dim)
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            create_activation(self.activation, self.hidden_dim)
        )

        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim if i == 0 else self.mlp_dim, eps=1e-6),
                ResNetDense(
                    input_dim=self.hidden_dim if i == 0 else self.mlp_dim,
                    hidden_dim=self.mlp_dim,
                    output_dim=self.mlp_dim,
                    num_layers=self.resnet_num_layer,
                    activation=self.activation,
                    dropout=self.dropout,
                    layer_scale_init=self.layer_scale_init
                )
            )
            for i in range(self.num_layer - 1)
        ])

        self.out_layer_norm = nn.LayerNorm(self.mlp_dim, eps=1e-6)
        self.out = nn.Linear(self.mlp_dim, output_dim)
        nn.init.zeros_(self.out.weight)

    def forward(self,
                x: Tensor,
                time: Tensor,
                x_mask: Optional[Tensor] = None,
                global_cond: Optional[Tensor] = None,
                label: Optional[Tensor] = None
                ) -> Tensor:
        # ----------------
        # x: [B, 1,] <- Noised Global Input
        # x_mask: [B, 1, 1] <- Mask
        # t: [B,] <- Time
        # global_cond: [B, 1, C] <- Global Condition
        # label: [B, 1] <- Conditional Label, one-hot in function
        # ----------------

        batch_size = x.shape[0]
        time = time.unsqueeze(-1)  # [B, 1]
        if x_mask is None:
            x_mask = torch.ones((batch_size, 1), device=x.device)

        x_mask = x_mask.reshape((batch_size, 1))

        embed_time = self.fourier_projection(time).unsqueeze(1)  # [B, 1, D]
        if global_cond is not None:
            global_cond_token = self.global_cond_embedding(global_cond[..., self.input_cond_indices])  # [B, 1, D]
            global_token = torch.cat([global_cond_token, embed_time], dim=-1)  # [B, 1, 2D]
        else:
            global_token = self.dense_t(embed_time)  # [B, 1, 2D]

        cond_token = self.cond_token_embedding(global_token)  # [B, 1, 2D]

        if label is not None:
            label = F.one_hot(label, num_classes=self.num_classes).to(
                dtype=self.label_embedding[0].weight.dtype)  # [B, 1, C]
            cond_label = self.label_embedding(label)  # [B, 1, 2D]
            cond_token = cond_token + cond_label

        scale, shift = torch.chunk(cond_token, 2, dim=-1)
        scale = scale.squeeze(1)  # [B, D]
        shift = shift.squeeze(1)  # [B, D]

        embed_x = self.dense_layer(x)
        embed_x = (embed_x * (1.0 + scale) + shift) * x_mask

        for resnet_layer in self.resnet_layers:
            embed_x = resnet_layer(embed_x) * x_mask
        embed_x = self.out_layer_norm(embed_x) * x_mask
        outputs = self.out(embed_x) * x_mask

        return outputs  # [B, output_dim]
