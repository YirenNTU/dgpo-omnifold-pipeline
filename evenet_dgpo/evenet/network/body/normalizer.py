import torch
from torch import nn, Tensor
import math
from torch.distributions import Normal
from typing import List


class Normalizer(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor, norm_mask: Tensor, inv_cdf_index=None, padding_size=0):

        super(Normalizer, self).__init__()

        if inv_cdf_index is None:
            inv_cdf_index = []
        """
        :param
            log_mask: mask to apply before normalization. shape (num_features,)
            mean: mean value for normalization. shape (num_features,)
            std: standard deviation for normalization . shape (num_features,)
        """
        # Initialize mean and std as parameters
        self.register_buffer("norm_mask", norm_mask)
        mean = torch.where(self.norm_mask, mean, 0.0)
        std = torch.where(self.norm_mask, std, 1.0)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp(min=1e-6))
        self.inv_cdf_index = inv_cdf_index
        self.normal = Normal(0, 1)
        self.padding = padding_size

        if self.padding > 0:
            # padding mean and std to fit the padding size
            self.mean = torch.cat([self.mean, torch.zeros(self.padding).to(self.mean.device)])
            self.std = torch.cat([self.std, torch.ones(self.padding).to(self.mean.device)])

        # self.log_mask_expanded = self.log_mask.unsqueeze(0).unsqueeze(0) if self.log_mask is not None else None

    @torch.no_grad()
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """
        # Apply the log mask to the input tensor
        # x = torch.where(self.log_mask_expanded, torch.log1p(x), x)  # log1p(x) = log(1 + x) to avoid log(0) issues # TODO
        x = (x - self.mean) / self.std
        if mask is not None:
            x = x * mask
        if len(self.inv_cdf_index) > 0:
            # After normalization, apply inverse CDF transformation
            x_partial = x[..., self.inv_cdf_index].contiguous()
            # normalized uniform: [-sqrt(3), sqrt(3)] , add extra 0.1 to avoid unperfect mean, std deviation
            # x_partial = (x_partial + (math.sqrt(3) + 0.1)) / (2 * (math.sqrt(3) + 0.1))
            # Yulei: Don't add extra 0.1
            x_partial = (x_partial + (math.sqrt(3))) / (2 * (math.sqrt(3)))
            x_partial = torch.clamp(x_partial, 1e-6, 1 - 1e-6)
            x[..., self.inv_cdf_index] = self.normal.icdf(x_partial)
            if mask is not None:
                x = x * mask

        return x

    @torch.no_grad()
    def denormalize(self, x: Tensor, mask: Tensor = None, remove_padding: bool = False, index: List = None) -> Tensor:
        """
        :param remove_padding: remove the padding part (for invisible generation)
        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :param index: index of features to apply inverse transformation
        :return: tensor (batch_size, num_objects, num_features)
        """
        return self._denormalize_impl(x, mask=mask, remove_padding=remove_padding, index=index)

    def denormalize_grad(self, x: Tensor, mask: Tensor = None, remove_padding: bool = False, index: List = None) -> Tensor:
        """Gradient-capable denormalize.

        Identical math to :meth:`denormalize` but **without** the ``@torch.no_grad()``
        guard, so the autograd graph is preserved. Required by paths that backprop
        through generated/predicted values into the policy network (e.g. the DGPO
        discriminator-Wasserstein projection constraint). Inference/rollout/eval code
        should keep using :meth:`denormalize`.
        """
        return self._denormalize_impl(x, mask=mask, remove_padding=remove_padding, index=index)

    def _denormalize_impl(self, x: Tensor, mask: Tensor = None, remove_padding: bool = False, index: List = None) -> Tensor:
        """Shared denormalize math. Written out-of-place so it stays autograd-safe
        when called with grad enabled (no in-place writes into a grad-tracked tensor)."""
        if remove_padding:
            current_mean = self.mean[:-self.padding]
            current_std = self.std[:-self.padding]
        else:
            current_mean = self.mean
            current_std = self.std

        if len(self.inv_cdf_index) > 0:
            if index is not None:
                inv_cdf_index = []
                for idx in index:
                    if idx in self.inv_cdf_index:
                        inv_cdf_index.append(idx)
            else:
                inv_cdf_index = self.inv_cdf_index

            x_partial = x[..., inv_cdf_index].contiguous()
            # ``Normal.cdf._validate_sample`` rejects any NaN/inf even on masked-out
            # slots; sanitize so a single bad row cannot abort the whole rollout.
            x_partial = torch.nan_to_num(x_partial, nan=0.0, posinf=8.0, neginf=-8.0)
            x_partial = self.normal.cdf(x_partial)
            # x_partial = x_partial * 2 * (math.sqrt(3) + 0.1) - (math.sqrt(3) + 0.1)
            # Yulei: Don't add extra 0.1
            x_partial = x_partial * 2 * (math.sqrt(3)) - (math.sqrt(3))
            # Out-of-place scatter into the inv-CDF feature columns (avoids the
            # in-place ``x[..., idx] = ...`` that breaks autograd under grad-enabled use).
            idx_tensor = torch.as_tensor(inv_cdf_index, device=x.device, dtype=torch.long)
            x = x.index_copy(-1, idx_tensor, x_partial.to(dtype=x.dtype))
            if mask is not None:
                x = x * mask

        if index is not None:
            x = x * current_std[index] + current_mean[index]
        else:
            x = (x * current_std) + current_mean
        # x = torch.where(self.log_mask_expanded, torch.expm1(x), x) # TODO
        if mask is not None:
            x = x * mask

        return x
