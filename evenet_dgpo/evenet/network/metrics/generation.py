from typing import Callable, Union

import numpy as np
import torch
from evenet.utilities.diffusion_sampler import DDIMSampler
from evenet.utilities.debug_tool import debug_nonfinite_batch
from functools import partial

import matplotlib.pyplot as plt
from evenet.network.loss.generation import loss as gen_loss
from evenet.utilities.debug_tool import time_decorator
from typing import Dict
import wandb
import copy
from scipy.spatial.distance import jensenshannon
import logging


logger = logging.getLogger(__name__)

class GenerationMetrics:
    def __init__(
            self, device, class_names,
            sequential_feature_names,
            invisible_feature_names,
            target_global_names, target_global_index, target_event_index,
            hist_xmin=-15, hist_xmax=15, num_bins=60,
            global_generation=False,
            point_cloud_generation=False,
            neutrino_generation=False,
            use_generation_result=False,
            special_bin_configs: dict[str, list] = None,
            coordinate_system: str = "pt_eta_phi",
    ):

        self.sampler = DDIMSampler(device)
        self.device = device

        self.global_generation = global_generation
        self.point_cloud_generation = point_cloud_generation
        self.neutrino_generation = neutrino_generation
        self.use_generation_result = use_generation_result
        self.coordinate_system = coordinate_system

        # Default values for histogram
        self.num_bins = num_bins
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax

        self.sequential_feature_names = sequential_feature_names
        self.invisible_feature_names = invisible_feature_names
        self.target_global_names = target_global_names
        self.target_global_index = target_global_index
        self.target_event_index = target_event_index

        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.num_classes = len(class_names)
        self.class_names = class_names

        self.histogram = dict()
        self.truth_histogram = dict()

        self.histogram_2d = dict()
        self.pearson_stats = dict()
        self.neutrino_pt_bins = np.linspace(0.0, 300.0, self.num_bins + 1)
        self.neutrino_rel_pt_bins = np.linspace(-1.5, 1.5, self.num_bins + 1)
        self.neutrino_profile_bins = {
            "pt": np.linspace(0.0, 300.0, 31),
            "eta": np.linspace(-4.0, 4.0, 31),
            "phi": np.linspace(-3.2, 3.2, 31),
            "px": np.linspace(-300.0, 300.0, 31),
            "py": np.linspace(-300.0, 300.0, 31),
            "pz": np.linspace(-300.0, 300.0, 31),
        }
        self.neutrino_pt_profile_bins = self.neutrino_profile_bins["pt"]
        self.neutrino_pt_histogram = dict()
        self.neutrino_truth_pt_histogram = dict()
        self.neutrino_rel_pt_histogram = dict()
        self.neutrino_delta_profiles = {
            name: dict() for name in self.neutrino_profile_bins
        }
        self.neutrino_pt_delta_profile = self.neutrino_delta_profiles["pt"]

        # Per-variable neutrino kinematic distributions (pred vs truth). These are
        # recovered from the generated (px, py, pz) via ``_neutrino_profile_values``
        # so that eta/phi/pt are always available (even in Cartesian mode) and
        # px/py/pz are histogrammed with physical (GeV) binning instead of the
        # generic [-15, 15] window used by ``self.bins``.
        self.neutrino_kin_vars = ["pt", "eta", "phi", "px", "py", "pz"]
        self.neutrino_pred_dist = {name: dict() for name in self.neutrino_kin_vars}
        self.neutrino_truth_dist = {name: dict() for name in self.neutrino_kin_vars}

        self.special_bins = dict()
        self.special_bins_centers = dict()
        if special_bin_configs is not None:
            for name, special_bins in special_bin_configs.items():
                self.special_bins[name] = np.linspace(special_bins[1], special_bins[2], special_bins[0])
                self.special_bins_centers[name] = 0.5 * (self.special_bins[name][:-1] + self.special_bins[name][1:])

    @time_decorator(name="[Generation] update metrics")
    def update(
            self,
            model,
            input_set,
            num_steps_global=20,
            num_steps_point_cloud=40,
            num_steps_neutrino=40,
            eta=1.0,
            schedules: Union[None, dict] = None,
    ):
        model.eval()

        predict_distribution = dict()
        truth_distribution = dict()
        process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
            input_set['conditions_mask']).long()  # (batch_size, 1)
        masking = dict()

        do_recon = True
        do_truth = True
        if schedules is not None:
            do_recon = schedules.get('generation', False)
            do_truth = schedules.get('neutrino_generation', False)

        if self.global_generation:
            ####################################
            ##  Step 1: Generate num vectors  ##
            ####################################

            predict_for_global = partial(
                model.predict_diffusion_vector,
                cond_x=input_set,
                mode="global"
            )

            data_shape = [input_set['num_sequential_vectors'].shape[0], 1 + len(self.target_global_names)]
            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_global,
                normalize_fn=None,
                num_steps=num_steps_global,
                eta=eta,
                use_tqdm=False,
                process_name=f"Global",
            )

            generated_num_sequential_vectors = generated_distribution[..., 0]
            generated_num_sequential_vectors = model.num_point_cloud_normalizer.denormalize(
                generated_num_sequential_vectors)

            predict_distribution["num_vectors"] = torch.floor(generated_num_sequential_vectors.flatten() + 0.5)
            truth_distribution["num_vectors"] = input_set['num_sequential_vectors'].flatten()

            if len(self.target_global_names) > 0:
                generated_global = generated_distribution[..., 1:]
                generated_global = model.global_normalizer.denormalize(generated_global, index=self.target_global_index)
                for idx, name in enumerate(self.target_global_names):
                    predict_distribution[f"global-{name}"] = generated_global[..., idx].flatten()
                    truth_distribution[f"global-{name}"] = (
                        input_set['conditions'][..., self.target_global_index[idx]]).flatten()

                if self.use_generation_result:
                    input_set = copy.deepcopy(input_set)
                    input_set['conditions'][..., self.target_global_index] = generated_global

        if self.point_cloud_generation and do_recon:
            ####################################
            ##  Step 2: Generate point cloud  ##
            ####################################

            data_shape = input_set['x'].shape
            process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
                input_set['conditions_mask']).long()  # (batch_size, 1)

            predict_for_point_cloud = partial(
                model.predict_diffusion_vector,
                mode="event",
                cond_x=input_set,
                noise_mask=input_set["x_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )  # TODO: add stuff from previous step.

            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_point_cloud,
                normalize_fn=model.sequential_normalizer,
                eta=eta,
                num_steps=num_steps_point_cloud,
                noise_mask=input_set["x_mask"].unsqueeze(-1),  # [B, T, 1] to match noise x
                use_tqdm=False,
                process_name=f"PointCloud",
            )

            for i in range(data_shape[-1]):
                if i in self.target_event_index:
                    masking[f"point cloud-{self.sequential_feature_names[i]}"] = input_set["x_mask"]
                    predict_distribution[f"point cloud-{self.sequential_feature_names[i]}"] = generated_distribution[
                        ..., i]
                    truth_distribution[f"point cloud-{self.sequential_feature_names[i]}"] = input_set['x'][..., i]


        if self.neutrino_generation and do_truth:
            #####################################
            ## Generate invisible point cloud  ##
            #####################################
            # When cartesian: use x_invisible_cartesian (px, py, pz) from parquet
            invisible_key = 'x_invisible_cartesian' if self.coordinate_system == "cartesian" else 'x_invisible'
            data_shape = input_set[invisible_key].shape
            process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
                input_set['conditions_mask'].flatten()).long()  # (batch_size, 1)

            predict_for_neutrino = partial(
                model.predict_diffusion_vector,
                mode="neutrino",
                cond_x=input_set,
                noise_mask=input_set["x_invisible_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )

            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_neutrino,
                normalize_fn=model.invisible_normalizer,
                eta=eta,
                num_steps=num_steps_neutrino,
                use_tqdm=False,
                process_name=f"Neutrino",
                remove_padding=True,
            )

            # In Cartesian mode the raw (px, py, pz) span hundreds of GeV, so the
            # generic [-15, 15] 1D/2D histograms are meaningless (they only capture
            # a flat sliver around 0, which is why they look "uniform"). Skip them
            # and rely on the dedicated, physically-binned kinematic distributions
            # produced in ``update_neutrino_pt_diagnostics`` below.
            if self.coordinate_system != "cartesian":
                for i in range(data_shape[-1]):
                    masking[f"neutrino-{self.invisible_feature_names[i]}"] = input_set["x_invisible_mask"]
                    predict_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = generated_distribution[..., i]
                    truth_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = input_set[invisible_key][..., i]

            self.update_neutrino_pt_diagnostics(
                generated_distribution=generated_distribution,
                truth_distribution=input_set[invisible_key],
                mask=input_set["x_invisible_mask"],
                process_id=process_id,
            )

        # --------------- working line -----------------
        for distribution_name, distribution in predict_distribution.items():

            num_bins = self.num_bins
            if distribution_name in self.special_bins:
                num_bins = len(self.special_bins_centers[distribution_name])

            if distribution_name not in self.histogram:
                self.histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }
            if distribution_name not in self.truth_histogram:
                self.truth_histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }

            if distribution_name not in self.histogram_2d:
                self.histogram_2d[distribution_name] = {
                    class_name: np.zeros((num_bins, num_bins))
                    for class_name in self.class_names
                }

            if distribution_name not in self.pearson_stats:
                self.pearson_stats[distribution_name] = {
                    class_name: {
                        'sum_x': 0.0, 'sum_y': 0.0,
                        'sum_xx': 0.0, 'sum_yy': 0.0,
                        'sum_xy': 0.0, 'n': 0
                    } for class_name in self.class_names
                }

            for class_index, class_name in enumerate(self.class_names):
                class_mask = (process_id == class_index)
                if distribution_name in masking and (
                        predict_distribution[distribution_name].size() == masking[distribution_name].size()):
                    # Masking for point cloud
                    total_mask = masking[distribution_name][class_mask].flatten()
                    pred = predict_distribution[distribution_name][class_mask].flatten()[
                        total_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten()[
                        total_mask].detach().cpu().numpy()
                else:
                    pred = predict_distribution[distribution_name][class_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten().detach().cpu().numpy()

                hist_bins = self.bins
                if distribution_name in self.special_bins:
                    hist_bins = self.special_bins[distribution_name]

                hist, _ = np.histogram(pred, bins=hist_bins)
                self.histogram[distribution_name][class_name] += hist

                hist, _ = np.histogram(truth, bins=hist_bins)
                self.truth_histogram[distribution_name][class_name] += hist

                hist2d, _, _ = np.histogram2d(pred, truth, bins=[hist_bins, hist_bins])
                self.histogram_2d[distribution_name][class_name] += hist2d

                # Pearson stats
                stats = self.pearson_stats[distribution_name][class_name]
                stats['sum_x'] += pred.sum()
                stats['sum_y'] += truth.sum()
                stats['sum_xx'] += (pred ** 2).sum()
                stats['sum_yy'] += (truth ** 2).sum()
                stats['sum_xy'] += (pred * truth).sum()
                stats['n'] += pred.shape[0]

    def reset(self):
        self.histogram = dict()
        self.truth_histogram = dict()

        self.histogram_2d = dict()
        self.pearson_stats = dict()
        self.neutrino_pt_histogram = dict()
        self.neutrino_truth_pt_histogram = dict()
        self.neutrino_rel_pt_histogram = dict()
        self.neutrino_delta_profiles = {
            name: dict() for name in self.neutrino_profile_bins
        }
        self.neutrino_pt_delta_profile = self.neutrino_delta_profiles["pt"]
        self.neutrino_pred_dist = {name: dict() for name in self.neutrino_kin_vars}
        self.neutrino_truth_dist = {name: dict() for name in self.neutrino_kin_vars}

    def _pt_from_neutrino_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return neutrino pT [GeV] from Cartesian or log-pT/eta/phi features."""
        if self.coordinate_system == "cartesian":
            return torch.sqrt(features[..., 0] * features[..., 0] + features[..., 1] * features[..., 1] + 1e-12)
        return torch.expm1(features[..., 0].clamp(-10.0, 10.0))

    def _neutrino_profile_values(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return physical variables used by DGPO-style truth-binned profiles."""
        if self.coordinate_system == "cartesian":
            px = features[..., 0]
            py = features[..., 1]
            pz = features[..., 2]
            pt = torch.sqrt(px * px + py * py + 1e-12)
            eta = torch.where(pt > 1e-8, torch.asinh(pz / pt), torch.zeros_like(pz))
            phi = torch.atan2(py, px)
        else:
            log_pt = features[..., 0].clamp(-10.0, 10.0)
            eta = features[..., 1]
            phi = features[..., 2]
            pt = torch.expm1(log_pt)
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)

        return {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "px": px,
            "py": py,
            "pz": pz,
        }

    @staticmethod
    def _profile_delta(profile_name: str, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Return signed residuals, wrapping phi into [-pi, pi]."""
        delta = pred - truth
        if profile_name == "phi":
            delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        return delta

    def _ensure_neutrino_pt_diagnostic_buffers(self, class_name: str):
        if class_name not in self.neutrino_pt_histogram:
            self.neutrino_pt_histogram[class_name] = np.zeros(len(self.neutrino_pt_bins) - 1, dtype=np.float64)
            self.neutrino_truth_pt_histogram[class_name] = np.zeros(len(self.neutrino_pt_bins) - 1, dtype=np.float64)
            self.neutrino_rel_pt_histogram[class_name] = np.zeros(len(self.neutrino_rel_pt_bins) - 1, dtype=np.float64)

        for profile_name, bins in self.neutrino_profile_bins.items():
            if class_name in self.neutrino_delta_profiles[profile_name]:
                continue
            n_profile = len(bins) - 1
            self.neutrino_delta_profiles[profile_name][class_name] = {
                "sum": np.zeros(n_profile, dtype=np.float64),
                "sum_sq": np.zeros(n_profile, dtype=np.float64),
                "count": np.zeros(n_profile, dtype=np.float64),
            }
        self.neutrino_pt_delta_profile = self.neutrino_delta_profiles["pt"]

        for var in self.neutrino_kin_vars:
            if class_name in self.neutrino_pred_dist[var]:
                continue
            n_dist = len(self.neutrino_profile_bins[var]) - 1
            self.neutrino_pred_dist[var][class_name] = np.zeros(n_dist, dtype=np.float64)
            self.neutrino_truth_dist[var][class_name] = np.zeros(n_dist, dtype=np.float64)

    def _accumulate_delta_profile(
            self,
            profile_name: str,
            class_name: str,
            pred_values: torch.Tensor,
            truth_values: torch.Tensor,
    ):
        delta_values = self._profile_delta(profile_name, pred_values, truth_values)
        truth = truth_values.detach().float().cpu().numpy()
        delta = delta_values.detach().float().cpu().numpy()
        finite = np.isfinite(truth) & np.isfinite(delta)
        if profile_name == "pt":
            finite &= truth >= 0.0
        truth = truth[finite]
        delta = delta[finite]
        if truth.size == 0:
            return

        bins = self.neutrino_profile_bins[profile_name]
        bin_idx = np.digitize(truth, bins) - 1
        valid_bins = (bin_idx >= 0) & (bin_idx < len(bins) - 1)
        if not np.any(valid_bins):
            return

        profile = self.neutrino_delta_profiles[profile_name][class_name]
        idx = bin_idx[valid_bins]
        vals = delta[valid_bins]
        profile["sum"] += np.bincount(idx, weights=vals, minlength=profile["sum"].size)
        profile["sum_sq"] += np.bincount(idx, weights=vals * vals, minlength=profile["sum_sq"].size)
        profile["count"] += np.bincount(idx, minlength=profile["count"].size)

    def _accumulate_kin_distribution(
            self,
            var: str,
            class_name: str,
            pred_values: torch.Tensor,
            truth_values: torch.Tensor,
    ):
        """Accumulate pred/truth distribution histograms for a kinematic variable."""
        bins = self.neutrino_profile_bins[var]
        pred = pred_values.detach().float().cpu().numpy()
        truth = truth_values.detach().float().cpu().numpy()
        pred = pred[np.isfinite(pred)]
        truth = truth[np.isfinite(truth)]
        if pred.size:
            hist, _ = np.histogram(pred, bins=bins)
            self.neutrino_pred_dist[var][class_name] += hist
        if truth.size:
            hist, _ = np.histogram(truth, bins=bins)
            self.neutrino_truth_dist[var][class_name] += hist

    def update_neutrino_pt_diagnostics(
            self,
            generated_distribution: torch.Tensor,
            truth_distribution: torch.Tensor,
            mask: torch.Tensor,
            process_id: torch.Tensor,
    ):
        """Accumulate neutrino histograms and DGPO-style truth-binned profiles."""
        pred_profiles = self._neutrino_profile_values(generated_distribution)
        truth_profiles = self._neutrino_profile_values(truth_distribution)
        pred_pt = pred_profiles["pt"]
        truth_pt = truth_profiles["pt"]

        slot_mask = mask
        if slot_mask.dim() == 3 and slot_mask.shape[-1] == 1:
            slot_mask = slot_mask.squeeze(-1)
        slot_mask = slot_mask > 0
        proc = process_id.reshape(-1)

        for class_index, class_name in enumerate(self.class_names):
            self._ensure_neutrino_pt_diagnostic_buffers(class_name)
            event_mask = proc == class_index
            if not bool(event_mask.any().item()):
                continue

            valid_slots = slot_mask[event_mask]
            pred = pred_pt[event_mask][valid_slots].detach().float().cpu().numpy()
            truth = truth_pt[event_mask][valid_slots].detach().float().cpu().numpy()
            finite = np.isfinite(pred) & np.isfinite(truth) & (truth >= 0.0)
            pred = pred[finite]
            truth = truth[finite]
            if pred.size == 0:
                continue

            pred_hist, _ = np.histogram(pred, bins=self.neutrino_pt_bins)
            truth_hist, _ = np.histogram(truth, bins=self.neutrino_pt_bins)
            self.neutrino_pt_histogram[class_name] += pred_hist
            self.neutrino_truth_pt_histogram[class_name] += truth_hist

            delta = pred - truth
            rel_mask = truth > 1e-6
            rel_pt = pred[rel_mask] / truth[rel_mask] - 1.0
            rel_pt = rel_pt[np.isfinite(rel_pt)]
            rel_hist, _ = np.histogram(rel_pt, bins=self.neutrino_rel_pt_bins)
            self.neutrino_rel_pt_histogram[class_name] += rel_hist

            for profile_name in self.neutrino_profile_bins:
                pred_values = pred_profiles[profile_name][event_mask][valid_slots]
                truth_values = truth_profiles[profile_name][event_mask][valid_slots]
                self._accumulate_delta_profile(
                    profile_name=profile_name,
                    class_name=class_name,
                    pred_values=pred_values,
                    truth_values=truth_values,
                )
                if profile_name in self.neutrino_kin_vars:
                    self._accumulate_kin_distribution(
                        var=profile_name,
                        class_name=class_name,
                        pred_values=pred_values,
                        truth_values=truth_values,
                    )

    def reduce_across_gpus(self):
        if not torch.distributed.is_initialized():
            return

        # Helper function to reduce a nested dict
        def reduce_nested_histogram(nested_hist, dtype=torch.long):
            for name_, hist_group in nested_hist.items():
                for class_name_, data in hist_group.items():
                    tensor_ = torch.tensor(data, dtype=dtype, device=self.device)
                    torch.distributed.all_reduce(tensor_, op=torch.distributed.ReduceOp.SUM)
                    nested_hist[name_][class_name_] = tensor_.cpu().numpy()

        def reduce_class_histogram(class_hist, dtype=torch.float32):
            for class_name_, data in class_hist.items():
                tensor_ = torch.tensor(data, dtype=dtype, device=self.device)
                torch.distributed.all_reduce(tensor_, op=torch.distributed.ReduceOp.SUM)
                class_hist[class_name_] = tensor_.cpu().numpy()

        reduce_nested_histogram(self.histogram, dtype=torch.long)
        reduce_nested_histogram(self.truth_histogram, dtype=torch.long)
        reduce_nested_histogram(self.histogram_2d, dtype=torch.long)
        reduce_class_histogram(self.neutrino_pt_histogram, dtype=torch.float32)
        reduce_class_histogram(self.neutrino_truth_pt_histogram, dtype=torch.float32)
        reduce_class_histogram(self.neutrino_rel_pt_histogram, dtype=torch.float32)

        for profiles in self.neutrino_delta_profiles.values():
            for class_name, profile in profiles.items():
                for key in ["sum", "sum_sq", "count"]:
                    tensor = torch.tensor(profile[key], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    profile[key] = tensor.cpu().numpy()

        for var in self.neutrino_kin_vars:
            reduce_class_histogram(self.neutrino_pred_dist[var], dtype=torch.float32)
            reduce_class_histogram(self.neutrino_truth_dist[var], dtype=torch.float32)

        for name, stats_group in self.pearson_stats.items():
            for class_name, stats in stats_group.items():
                for key in ['sum_x', 'sum_y', 'sum_xx', 'sum_yy', 'sum_xy', 'n']:
                    tensor = torch.tensor(stats[key], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    stats[key] = tensor.item()

    def plot_histogram_func(
            self,
            truth_histogram,
            histogram,
            bin_widths,
            bin_centers,
    ):

        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        fig, ax = plt.subplots()

        jsd = dict()
        for cls, cls_name in enumerate(self.class_names):
            # Plot training histogram (bars)
            counts = histogram[cls_name]
            if np.sum(counts) > 0:
                density = counts / (np.sum(counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Pred)"
                plt.plot(
                    bin_centers,
                    density,
                    color=color,
                    label=label,
                    linestyle='--',
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
            truth_counts = truth_histogram[cls_name]
            if np.sum(truth_counts) > 0:
                truth_density = truth_counts / (np.sum(truth_counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Truth)"
                plt.bar(
                    bin_centers,
                    truth_density,
                    width=bin_widths,
                    color=color,
                    alpha=0.7,
                    label=f"{cls_name} (Truth)", edgecolor=color, fill=False
                )

            if (np.sum(counts) > 0) and (np.sum(truth_counts) > 0):
                p = truth_counts / np.sum(truth_counts)
                q = counts / np.sum(counts)
                jsd[cls_name] = (jensenshannon(p, q))

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        # plt.show()

        return fig, jsd

    def plot_histogram2d_func(self, histogram2d, x_centers, y_centers, title="2D Histogram"):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        pcm = ax.pcolormesh(X, Y, histogram2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label="Counts")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')
        ax.set_title(title)
        return fig

    def plot_neutrino_pt_histogram_func(self):
        """Overlay generated and truth neutrino pT histograms."""
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        centers = 0.5 * (self.neutrino_pt_bins[:-1] + self.neutrino_pt_bins[1:])
        widths = np.diff(self.neutrino_pt_bins)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]
        jsd = dict()
        for cls, class_name in enumerate(self.class_names):
            if class_name not in self.neutrino_pt_histogram:
                continue
            pred_counts = self.neutrino_pt_histogram[class_name]
            truth_counts = self.neutrino_truth_pt_histogram[class_name]
            color = colors[cls % len(colors)]
            if np.sum(pred_counts) > 0:
                pred_density = pred_counts / (np.sum(pred_counts) * widths)
                ax.plot(
                    centers,
                    pred_density,
                    color=color,
                    linestyle="--",
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=f"{class_name} pred",
                )
            if np.sum(truth_counts) > 0:
                truth_density = truth_counts / (np.sum(truth_counts) * widths)
                ax.bar(
                    centers,
                    truth_density,
                    width=widths,
                    color=color,
                    alpha=0.7,
                    edgecolor=color,
                    fill=False,
                    label=f"{class_name} truth",
                )
            if np.sum(pred_counts) > 0 and np.sum(truth_counts) > 0:
                p = truth_counts / np.sum(truth_counts)
                q = pred_counts / np.sum(pred_counts)
                jsd[class_name] = jensenshannon(p, q)
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel("Normalized density")
        ax.set_title("Neutrino pT distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig, jsd

    def plot_neutrino_rel_pt_histogram_func(self):
        """Plot the relative pT residual distribution, matching DGPO diagnostics."""
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        centers = 0.5 * (self.neutrino_rel_pt_bins[:-1] + self.neutrino_rel_pt_bins[1:])
        widths = np.diff(self.neutrino_rel_pt_bins)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]
        for cls, class_name in enumerate(self.class_names):
            counts = self.neutrino_rel_pt_histogram.get(class_name)
            if counts is None or np.sum(counts) <= 0:
                continue
            density = counts / (np.sum(counts) * widths)
            mean = np.sum(centers * counts) / np.sum(counts)
            ax.plot(
                centers,
                density,
                color=colors[cls % len(colors)],
                linestyle="-",
                linewidth=2,
                label=f"{class_name}: mean={mean:+.3f}",
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
        ax.set_ylabel("Normalized density")
        ax.set_title("Relative pT residual distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_neutrino_kin_distribution_func(self, var: str):
        """Overlay generated vs truth neutrino distribution for a kinematic variable.

        In Cartesian mode the predicted (px, py, pz) are recovered into
        (pt, eta, phi) via ``_neutrino_profile_values`` before accumulation, so
        eta/phi distributions are available and px/py/pz use physical GeV bins.
        """
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bins = self.neutrino_profile_bins[var]
        centers = 0.5 * (bins[:-1] + bins[1:])
        widths = np.diff(bins)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]
        jsd = dict()
        for cls, class_name in enumerate(self.class_names):
            pred_counts = self.neutrino_pred_dist[var].get(class_name)
            truth_counts = self.neutrino_truth_dist[var].get(class_name)
            color = colors[cls % len(colors)]
            if pred_counts is not None and np.sum(pred_counts) > 0:
                density = pred_counts / (np.sum(pred_counts) * widths)
                ax.plot(
                    centers, density, color=color, linestyle="--", marker="o",
                    linewidth=2, markersize=4, label=f"{class_name} (Pred)",
                )
            if truth_counts is not None and np.sum(truth_counts) > 0:
                truth_density = truth_counts / (np.sum(truth_counts) * widths)
                ax.bar(
                    centers, truth_density, width=widths, color=color, alpha=0.7,
                    edgecolor=color, fill=False, label=f"{class_name} (Truth)",
                )
            if (pred_counts is not None and truth_counts is not None
                    and np.sum(pred_counts) > 0 and np.sum(truth_counts) > 0):
                p = truth_counts / np.sum(truth_counts)
                q = pred_counts / np.sum(pred_counts)
                jsd[class_name] = jensenshannon(p, q)
        unit = " [GeV]" if var in {"pt", "px", "py", "pz"} else (" [rad]" if var == "phi" else "")
        ax.set_xlabel(f"{var}{unit}")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"Neutrino {var} distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig, jsd

    @staticmethod
    def _profile_axis_labels(profile_name: str) -> tuple[str, str, str]:
        display = {
            "pt": "pT",
            "eta": "eta",
            "phi": "phi",
            "px": "px",
            "py": "py",
            "pz": "pz",
        }.get(profile_name, profile_name)
        if profile_name in {"pt", "px", "py", "pz"}:
            return f"Truth {display} [GeV]", f"Mean delta {display} [GeV]", display
        if profile_name == "phi":
            return "Truth phi [rad]", "Mean wrapped delta phi [rad]", display
        return f"Truth {display}", f"Mean delta {display}", display

    def plot_neutrino_delta_profile_func(self, profile_name: str):
        """Plot mean ``pred - truth`` in truth-variable bins, DGPO-style."""
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bins = self.neutrino_profile_bins[profile_name]
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = float(bins[1] - bins[0])
        total_counts = np.zeros_like(centers)
        x_label, y_label, display = self._profile_axis_labels(profile_name)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]
        for cls, class_name in enumerate(self.class_names):
            profile = self.neutrino_delta_profiles[profile_name].get(class_name)
            if profile is None:
                continue
            counts = profile["count"]
            valid = counts > 0
            total_counts += counts
            if not np.any(valid):
                continue
            means = np.full_like(centers, np.nan, dtype=np.float64)
            errors = np.full_like(centers, np.nan, dtype=np.float64)
            means[valid] = profile["sum"][valid] / counts[valid]
            variance = np.maximum(profile["sum_sq"][valid] / counts[valid] - means[valid] ** 2, 0.0)
            errors[valid] = np.sqrt(variance / counts[valid])
            ax.errorbar(
                centers[valid],
                means[valid],
                yerr=errors[valid],
                fmt="o-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=colors[cls % len(colors)],
                label=f"{class_name} mean delta {display}",
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Neutrino {display} bias vs truth {display}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        ax_count = ax.twinx()
        ax_count.bar(centers, total_counts, width=width * 0.85, alpha=0.12, color="gray", label="entries")
        ax_count.set_ylabel("Entries")
        fig.tight_layout()
        return fig

    def plot_neutrino_pt_delta_profile_func(self):
        """Plot mean ``pT_pred - pT_truth`` in truth-pT bins, DGPO-style."""
        return self.plot_neutrino_delta_profile_func("pt")

    def plot_histogram(self):
        figs = dict()

        jsd_results = dict()
        for name in self.histogram:
            bin_centers = self.bin_centers
            bin_widths = np.diff(self.bins)

            if name in self.special_bins:
                bin_widths = np.diff(self.special_bins[name])
                bin_centers = self.special_bins_centers[name]

            figs[f"{name}-1d"], jsd = self.plot_histogram_func(
                self.truth_histogram[name],
                self.histogram[name],
                bin_widths=bin_widths,
                bin_centers=bin_centers,
            )
            for cls_name, score in jsd.items():
                jsd_results[f"{name}-{cls_name}"] = score

            for class_name in self.class_names:
                if class_name not in jsd:
                    continue
                if 'neutrino' not in name:
                    continue

                fig = self.plot_histogram2d_func(
                    self.histogram_2d[name][class_name],
                    x_centers=bin_centers,
                    y_centers=bin_centers,
                    title=f"2D Histogram {name} - {class_name}"
                )
                figs[f"2D_{name}_{class_name}"] = fig

        if self.neutrino_pt_histogram:
            fig_pt, jsd_pt = self.plot_neutrino_pt_histogram_func()
            figs["neutrino-pt-1d"] = fig_pt
            for cls_name, score in jsd_pt.items():
                jsd_results[f"neutrino-pt-{cls_name}"] = score

        if self.neutrino_rel_pt_histogram:
            figs["neutrino-rel_pt-1d"] = self.plot_neutrino_rel_pt_histogram_func()

        for var in self.neutrino_kin_vars:
            has_pred = any(np.sum(c) > 0 for c in self.neutrino_pred_dist[var].values())
            has_truth = any(np.sum(c) > 0 for c in self.neutrino_truth_dist[var].values())
            if not (has_pred or has_truth):
                continue
            fig_var, jsd_var = self.plot_neutrino_kin_distribution_func(var)
            figs[f"neutrino-{var}-dist-1d"] = fig_var
            for cls_name, score in jsd_var.items():
                jsd_results[f"neutrino-{var}-dist-{cls_name}"] = score

        for profile_name, profiles in self.neutrino_delta_profiles.items():
            if profiles:
                figs[f"neutrino-profile/{profile_name}_delta_vs_truth_{profile_name}"] = (
                    self.plot_neutrino_delta_profile_func(profile_name)
                )

        if self.neutrino_pt_delta_profile:
            figs["neutrino-pt-delta_vs_truth_pt"] = self.plot_neutrino_pt_delta_profile_func()

        # Pearson correlation
        pearson_results = dict()
        for name in self.pearson_stats:
            if 'neutrino' not in name:
                continue

            pearson_results[name] = dict()
            for class_name in self.class_names:

                if class_name not in jsd_results:
                    continue

                stats = self.pearson_stats[name][class_name]
                n = stats['n']
                numerator = n * stats['sum_xy'] - stats['sum_x'] * stats['sum_y']
                denominator = np.sqrt(
                    (n * stats['sum_xx'] - stats['sum_x'] ** 2) *
                    (n * stats['sum_yy'] - stats['sum_y'] ** 2)
                )
                if denominator == 0:
                    r = 0.0
                else:
                    r = numerator / denominator
                pearson_results[name][class_name] = r

        return figs, pearson_results, jsd_results


@time_decorator(name="[Generation] shared_step")
def shared_step(
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        gen_metrics: GenerationMetrics,
        model: torch.nn.Module,
        global_loss_scale: float,
        event_loss_scale: float,
        invisible_loss_scale: float,
        device: torch.device,
        loss_head_dict: dict,
        num_steps_global=20,
        num_steps_point_cloud=100,
        num_steps_neutrino=100,
        diffusion_on: bool = False,
        invisible_padding: int = 0,
        update_metric: bool = True,
        event_weight: torch.Tensor = None,
        schedules: Union[None, dict] = None
):
    generation_loss = dict()

    global_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    recon_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    truth_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    for generation_target, generation_result in outputs.items():
        feature_dim = generation_result["vector"].shape[-1]
        if generation_target == "point_cloud":
            masking = generation_result["mask"]
        elif generation_target == "neutrino":
            masking = generation_result["mask"] # (B, N, 1)

            if invisible_padding > 0:
                B, N, _ = masking.shape
                # Expand to (B, N, F), with True for real features, False for padded ones
                masking = masking.expand(B, N, feature_dim).clone()
                masking[:, :, -invisible_padding:] = False  # mask out padded features
                feature_dim = 1
        elif generation_target == "num_point_cloud":
            masking = None
            feature_dim = None
        else:
            masking = None
            feature_dim = None

        generation_loss[generation_target] = gen_loss(
            predict=generation_result["vector"],
            target=generation_result["truth"],
            mask=masking,
            feature_dim=feature_dim,
            event_weight=event_weight
        )

        debug_nonfinite_batch(
            {
                "predict": generation_result["vector"],
                "truth": generation_result["truth"],
                "mask": masking,
                "weight": event_weight,
            },
            batch_dim=0,  # change if your batch axis differs
            name=f"gen/{generation_target}",
            logger=logger,
        )

        # if generation loss is nan, then print all details
        if torch.isnan(generation_loss[generation_target]):
            logger.warning(
                f"NaN in generation loss for {generation_target} "
                f"predict: {generation_result['vector']}, truth: {generation_result['truth']}, "
            )

            generation_loss[generation_target] = 0.0

        if generation_target == "global":
            global_gen_loss = global_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-global"] = global_gen_loss
        elif generation_target == "neutrino":
            truth_gen_loss = truth_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-truth"] = truth_gen_loss

        elif generation_target == "point_cloud":
            recon_gen_loss = recon_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-recon"] = recon_gen_loss

        if diffusion_on and update_metric:
            gen_metrics.update(
                model=model,
                input_set=batch,
                num_steps_global=num_steps_global,
                num_steps_point_cloud=num_steps_point_cloud,
                num_steps_neutrino=num_steps_neutrino,
                schedules=schedules,
            )

    loss = (global_gen_loss * global_loss_scale + recon_gen_loss * event_loss_scale + truth_gen_loss * invisible_loss_scale) / len(
        outputs)
    # print(f"Training: {model.training}, loss scale: {event_loss_scale}, total sum: {torch.sum(masking) if masking is not None else masking}, Global loss: {global_gen_loss.item()}, Recon loss: {recon_gen_loss.item()}, Truth loss: {truth_gen_loss.item()}, loss: {loss.item()}")

    return loss, generation_loss


@time_decorator(name="[Generation] shared_epoch_end")
def shared_epoch_end(
        global_rank,
        metrics_valid: GenerationMetrics,
        metrics_train: GenerationMetrics,
        logger,
):
    metrics_valid.reduce_across_gpus()
    if metrics_train:
        metrics_train.reduce_across_gpus()

    if global_rank == 0:
        category_map = {
            "neutrino-": "generation-invisible",
            "point cloud-": "generation-event",
            "global-": "generation-global"
        }
        figs, extra, jsd_results = metrics_valid.plot_histogram()
        for name, fig in figs.items():

            for prefix, category in category_map.items():
                if prefix in name:
                    tag = name.replace(prefix, "")
                    logger.log({f"{category}/{tag}": wandb.Image(fig)})
                    break

            plt.close(fig)

        for name in extra:
            for class_name, value in extra[name].items():
                # logger.log({f"generation/pearson_{name}_{class_name}": value})

                for prefix, category in category_map.items():
                    if prefix in name:
                        tag = name.replace(prefix, "")
                        logger.log({f"{category}/pearson/{tag}_{class_name}": value})
                        break

        for _ in jsd_results:
            for jsd_name, jsd_score in jsd_results.items():
                # logger.log({f"generation/jsd_{jsd_name}": jsd_score})

                for prefix, category in category_map.items():
                    if prefix in jsd_name:
                        tag = jsd_name.replace(prefix, "")
                        logger.log({f"{category}/jsd/{tag}": jsd_score})
                        break

    metrics_valid.reset()
    if metrics_train:
        metrics_train.reset()
