"""Fixed-panel, paired-noise coverage validation of the current raw policy."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


def diagnostic_helpers():
    # Reuse the same physics reconstruction and event bootstrap as h4ddim1.
    scripts = str(Path(__file__).resolve().parents[3] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import diagnose_h4_ddim_coverage as helpers
    return helpers


def due(epoch, every):
    return (epoch + 1) % every == 0


def event_noise(positions, k, seed):
    return torch.stack([torch.randn(k, 2, 2,
        generator=torch.Generator().manual_seed(seed + int(i))) for i in positions])


def numeric_metrics(result):
    out = {}
    for row in result['coverage']['regions']:
        prefix = f"{row['region']}/{row['threshold_radians']:g}"
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != 'threshold_radians':
                out[f'{prefix}/{key}'] = value
    for family in ('topology', 'target_marginals'):
        for name, values in result[family].items():
            out[f'{family}/{name}/w1_radians'] = values['w1_radians']
            if 'cdf' in values:
                c = values['cdf']
                out[f'{family}/{name}/cdf_max_gap'] = float(np.max(np.abs(np.array(c['truth']) - c['generated'])))
    for name, value in result['invalid_direction_inputs'].items():
        out[f'invalid/{name}'] = value
    return out


class JointCoverageValidation(Callback):
    def __init__(self, config, output):
        super().__init__()
        self.cfg = dict(config)
        self.output = Path(output) / 'joint_coverage'
        self.baseline = None
        self.baseline_metrics = None
        self.last_epoch = -1
        self.panel = None
        if int(self.cfg['every_n_epochs']) < 1 or int(self.cfg['K']) < 1:
            raise ValueError('Coverage cadence and K must be positive')

    def state_dict(self):
        return dict(baseline=self.baseline, baseline_metrics=self.baseline_metrics,
                    last_epoch=self.last_epoch, config=self.cfg)

    def load_state_dict(self, state):
        # Extending the endpoint changes evaluation timing, not panel/sampling.
        protocol = lambda cfg: {k: v for k, v in cfg.items() if k != 'final_completed_epoch'}
        if protocol(state['config']) != protocol(self.cfg):
            raise ValueError('Cannot resume paired coverage with a changed panel/protocol')
        self.baseline = state['baseline']
        self.baseline_metrics = state['baseline_metrics']
        self.last_epoch = state['last_epoch']

    def on_train_start(self, trainer, pl_module):
        # Lightning has loaded/synchronized the policy before this hook.
        if self.baseline is None:
            self.evaluate(trainer, pl_module, int(trainer.current_epoch))

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        scheduled = due(epoch, int(self.cfg['every_n_epochs'])) or epoch + 1 == self.cfg.get('final_completed_epoch')
        if not trainer.sanity_checking and scheduled and epoch != self.last_epoch:
            self.evaluate(trainer, pl_module, epoch + 1)
            self.last_epoch = epoch

    @torch.no_grad()
    def evaluate(self, trainer, module, completed_epochs):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        h = diagnostic_helpers()
        rank, world = trainer.global_rank, trainer.world_size
        if self.panel is None:
            self.panel = torch.load(self.cfg['panel_path'], map_location='cpu', weights_only=True)
            n = len(self.panel['truth'])
            if n != int(self.cfg['events']) or n < world:
                raise ValueError(f'Coverage panel has {n} events; expected {self.cfg["events"]}')
            for key in ('test_rows', 'pool_rows'):
                if len(self.panel[key].unique()) != n:
                    raise ValueError('Coverage panel contains duplicate event identities')
        panel = self.panel
        ids = torch.arange(rank, len(panel['truth']), world)
        spec = EventPackingSpec.from_dict(panel['packing_spec'])
        device = module.device
        model = module.model
        model_dtype = next(model.parameters()).dtype
        modes = [(m, m.training) for m in model.modules()]
        model.eval()
        outputs = []
        started = time.perf_counter()
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        try:
            # Validation must not change the subsequent training RNG stream.
            with torch.random.fork_rng(devices=devices):
                for start in range(0, len(ids), int(self.cfg['batch_size'])):
                    positions = ids[start:start + int(self.cfg['batch_size'])]
                    batch = unpack_event_inputs(panel['condition'][positions].to(device), spec)
                    batch = {key: value.to(dtype=model_dtype) if isinstance(value, torch.Tensor)
                             and value.is_floating_point() else value for key, value in batch.items()}
                    batch['x_invisible'] = torch.zeros(len(positions), 2, 2, device=device, dtype=model_dtype)
                    batch['x_invisible_mask'] = torch.ones(len(positions), 2, dtype=torch.bool, device=device)
                    # Event-specific CPU streams make noise independent of rank/world partition.
                    bank = event_noise(positions, int(self.cfg['K']), int(self.cfg['seed']))
                    sampler = h.make_replay_sampler(bank.permute(1, 0, 2, 3).to(device=device, dtype=model_dtype), self.cfg['x0_mode'])
                    generated = generate_neutrino_candidates(model, batch, sampler,
                        K=int(self.cfg['K']), num_ddim_steps=int(self.cfg['ddim_steps']),
                        device=device, parallel_chains=1)
                    outputs.append(generated.permute(1, 0, 2, 3).reshape(len(positions), int(self.cfg['K']), 4).cpu())
                    if rank == 0:
                        print(f'[joint coverage] completed_epochs={completed_epochs} rank0 events={start+len(positions)}/{len(ids)}', flush=True)
        finally:
            for m, mode in modes:
                m.training = mode
        part = dict(positions=ids, generated=torch.cat(outputs))
        parts = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(parts, part)
        else:
            parts = [part]
        generated = torch.empty(len(panel['truth']), int(self.cfg['K']), 4, dtype=model_dtype)
        for shard in parts:
            generated[shard['positions']] = shard['generated']
        if rank == 0:
            result, truth_angles, angles = h.analyze_arm(panel, generated)
            metrics = numeric_metrics(result)
            if self.baseline is None:
                self.baseline = torch.from_numpy(angles.copy())
                self.baseline_metrics = metrics.copy()
            comparison = h.paired_comparison(truth_angles, self.baseline.numpy(), angles,
                bootstrap=int(self.cfg.get('bootstrap', 500)), seed=int(self.cfg['seed']))
            for row in comparison:
                if row['region'] == 'joint':
                    prefix = f'paired/joint/{row["threshold_radians"]:g}'
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and key != 'threshold_radians':
                            metrics[f'{prefix}/{key}'] = value
                        elif key.endswith('ci95'):
                            metrics[f'{prefix}/{key}_lo'], metrics[f'{prefix}/{key}_hi'] = value
            for key, value in self.baseline_metrics.items():
                if key.endswith(('w1_radians', 'cdf_max_gap')):
                    metrics[f'change_from_baseline/{key}'] = metrics[key] - value
            self.output.mkdir(parents=True, exist_ok=True)
            report = dict(config=self.cfg, completed_epochs=completed_epochs,
                          global_step=int(trainer.global_step), result=result, paired=comparison)
            path = self.output / f'epoch-{completed_epochs:04d}.json'
            path.write_text(json.dumps(report, indent=2, allow_nan=False))
            torch.save(dict(generated=generated, angles=torch.from_numpy(angles)),
                       self.output / f'epoch-{completed_epochs:04d}.pt')
            metrics.update(completed_epochs=completed_epochs, seconds=time.perf_counter()-started,
                           events=len(panel['truth']), K=int(self.cfg['K']))
            for logger in trainer.loggers:
                logger.log_metrics({f'joint_coverage/{k}': v for k, v in metrics.items()}, step=trainer.global_step)
            self.log_plots(trainer, result, truth_angles, angles, completed_epochs)
        if dist.is_initialized():
            state = [self.baseline, self.baseline_metrics]
            dist.broadcast_object_list(state, src=0)
            self.baseline, self.baseline_metrics = state

    def log_plots(self, trainer, result, truth, generated, epoch):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        c = result['topology']['joint_radius']['cdf']
        axes[0].semilogx(c['threshold_radians'][1:], c['truth'][1:], label='Truth')
        axes[0].semilogx(c['threshold_radians'][1:], c['generated'][1:], label='Generated')
        axes[0].set(xlabel='Joint radius (rad)', ylabel='CDF', title=f'Completed epochs: {epoch}')
        axes[0].legend()
        bins = np.linspace(-8, np.log10(np.pi), 65)
        for ax, values, name in zip(axes[1:], [truth, generated.reshape(-1, 2)], ['Truth', 'Generated']):
            hist, _, _ = np.histogram2d(*np.log10(np.maximum(values, 1e-8)).T, bins=(bins, bins))
            ax.imshow((hist / len(values)).T, origin='lower', extent=[bins[0], bins[-1]]*2,
                      aspect='auto', norm=LogNorm(vmin=1e-5, vmax=1))
            ax.set(title=name, xlabel='log10 acoplanarity (rad)', ylabel='log10 acollinearity (rad)')
        fig.tight_layout()
        fig.savefig(self.output / f'epoch-{epoch:04d}.png')
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(key='joint_coverage/plots', images=[wandb.Image(fig)], step=trainer.global_step)
        plt.close(fig)
