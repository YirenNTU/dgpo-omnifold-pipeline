"""Read-only full-panel spin diagnostics inside the supervised 16-GPU group."""
from pathlib import Path
import json

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


class TauPhysicsValidation(Callback):
    def __init__(self, config, output):
        self.cfg = dict(config)
        self.output = Path(output).parent / 'tau_physics'
        self.panel = None
        self.baseline = None
        self.last_epoch = None
        if (int(self.cfg['events']) != 119002 or int(self.cfg['K']) != 8
                or int(self.cfg['ddim_steps']) != 20
                or min(int(self.cfg['every_n_epochs']), int(self.cfg['batch_size'])) < 1):
            raise ValueError('Tau physics requires full filtered validation, K8 and DDIM20')

    def state_dict(self):
        return dict(config=self.cfg, baseline=self.baseline, last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        if state['config'] != self.cfg:
            raise ValueError('Cannot resume tau physics with a changed evaluation protocol')
        self.baseline, self.last_epoch = state['baseline'], state['last_epoch']

    def on_train_start(self, trainer, pl_module):
        if self.baseline is None:
            self.evaluate(trainer, pl_module, int(trainer.current_epoch))

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if (not trainer.sanity_checking and epoch != self.last_epoch
                and epoch % int(self.cfg['every_n_epochs']) == 0):
            self.evaluate(trainer, pl_module, epoch)

    def on_train_end(self, trainer, pl_module):
        # A validation early-stop still receives a terminal physics diagnostic.
        epoch = int(trainer.current_epoch)
        if epoch != self.last_epoch:
            self.evaluate(trainer, pl_module, epoch)

    @torch.no_grad()
    def evaluate(self, trainer, module, epoch):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
        from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
        from RL.DGPO_neutrino.tau_reward_transfer import paired_statistics
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        from evenet.utilities.joint_coverage import event_noise
        from evenet.utilities.diffusion_sampler import make_replay_sampler

        rank, world = trainer.global_rank, trainer.world_size
        if world != 16:
            raise ValueError('Full-panel tau physics requires the training sixteen-GPU group')
        if self.panel is None:
            with np.load(self.cfg['panel_path'], allow_pickle=False) as f:
                self.panel = {k: f[k] for k in f.files}
            metadata = json.loads(Path(self.cfg['panel_metadata']).read_text())
            if (metadata['events'] != self.cfg['events']
                    or len(self.panel['source_ids']) != self.cfg['events']
                    or len(np.unique(self.panel['source_ids'])) != self.cfg['events']):
                raise ValueError('Unaligned or incomplete tau physics panel')
            self.spec = EventPackingSpec.from_dict(metadata['packing_spec'])
        panel, model, device = self.panel, module.model, module.device
        dtype = next(model.parameters()).dtype
        ids = np.arange(rank, len(panel['source_ids']), world)
        folder = self.output / f'epoch-{epoch:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        modes = [(m, m.training) for m in model.modules()]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        generated, terms = [], []
        try:
            model.eval()
            # Evaluation is isolated from subsequent dropout/noise streams.
            with torch.random.fork_rng(devices=devices):
                for start in range(0, len(ids), int(self.cfg['batch_size'])):
                    ii = ids[start:start + int(self.cfg['batch_size'])]
                    batch = unpack_event_inputs(torch.as_tensor(panel['raw_condition'][ii], device=device, dtype=dtype), self.spec)
                    # Preserve supervised labels, rather than the RL label0 contract.
                    batch['classification'] = batch['classification'].long()
                    batch['x_invisible'] = torch.zeros(len(ii), 2, 2, device=device, dtype=dtype)
                    batch['x_invisible_mask'] = torch.ones(len(ii), 2, device=device, dtype=torch.bool)
                    batch['event_category'] = torch.as_tensor(panel['category'][ii], device=device)
                    for leg in ('a', 'b'):
                        for j, key in enumerate(('E', 'px', 'py', 'pz')):
                            batch[f'lead_{leg}_visible_{key}'] = torch.as_tensor(panel['visible_' + leg][ii, j], device=device, dtype=dtype)
                    noise = event_noise(torch.as_tensor(ii), self.cfg['K'], self.cfg['seed'])
                    sampler = make_replay_sampler(noise.permute(1, 0, 2, 3).to(device=device, dtype=dtype), 'legacy')
                    draws = generate_neutrino_candidates(model, batch, sampler, K=self.cfg['K'],
                        num_ddim_steps=self.cfg['ddim_steps'], device=device, parallel_chains=1)
                    array = draws.permute(1, 0, 2, 3).cpu().numpy()
                    if not np.isfinite(array).all():
                        raise FloatingPointError('Nonfinite tau physics samples; no event dropping')
                    generated.append(array)
                    terms.append(np.mean([cij_terms(panel, ii, d) for d in array.swapaxes(0, 1)], axis=0))
                    if rank == 0:
                        print(f'[tau physics] epoch={epoch} rank0 events={start + len(ii)}/{len(ids)} K8', flush=True)
        finally:
            for m, mode in modes:
                m.training = mode
        np.savez(folder / f'rank-{rank:02d}.npz', positions=ids,
                 deltas=np.concatenate(generated), cij=np.concatenate(terms))
        dist.barrier()
        if rank == 0:
            n = len(panel['source_ids'])
            current = np.empty((n, 9), dtype=np.float64)
            draws = np.empty((n, self.cfg['K'], 2, 2), dtype=np.float32)
            seen = np.zeros(n, dtype=int)
            for other in range(world):
                with np.load(folder / f'rank-{other:02d}.npz', allow_pickle=False) as f:
                    pos = f['positions']
                    np.add.at(seen, pos, 1)
                    current[pos], draws[pos] = f['cij'], f['deltas']
            if not np.all(seen == 1) or not np.isfinite(current).all():
                raise ValueError('Missing, duplicate or nonfinite physics contributions')
            truth = cij_terms(panel, np.arange(n), panel['truth_deltas'])
            if self.baseline is None:
                self.baseline = current.copy()
            # Shared existing event bootstrap; dummy zero reward adds no objective.
            zero = np.zeros((n, 1))
            result = paired_statistics(zero, zero, self.baseline, current, truth,
                panel['event_weight'], replicates=self.cfg['bootstrap'], seed=self.cfg['seed'])['cij']
            from scripts.tau_tihsu_cij import panel_report, monitoring_metrics
            tihsu_report = panel_report(panel, draws)
            (folder / 'tihsu_reconstruction_cij.json').write_text(
                json.dumps(tihsu_report, indent=2, allow_nan=False) + '\n')
            report = dict(completed_epochs=epoch, global_step=int(trainer.global_step),
                events=n, candidates=self.cfg['K'], config=self.cfg, cij=result,
                legacy_cij_protocol='uncalibrated-event-moments-krn',
                tihsu_reconstruction=tihsu_report,
                scope='Raw samples, no classifier weights; existing fixed-energy tau reconstruction, not full rest-frame closure',
                uncertainty='Paired condition bootstrap, fixed generation noise; excludes training-seed uncertainty')
            (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            np.savez_compressed(folder / 'measurements.npz', source_ids=panel['source_ids'],
                deltas=draws, generated_cij=current, truth_cij=truth, weight=panel['event_weight'])
            metrics = {f'tau/physics/cij/{key}': result[key]
                       for key in ('error', 'diagonal_error', 'offdiagonal_error', 'nn_error')}
            metrics.update(monitoring_metrics(tihsu_report, 'tau/physics/tihsu_reconstruction/cij'))
            axes = ('k', 'r', 'n')
            for family in ('truth', 'generated', 'delta'):
                for i, left in enumerate(axes):
                    for j, right in enumerate(axes):
                        metrics[f'tau/physics/cij/{family}/{left}{right}'] = result[family][i][j]
            metrics.update({f'tau/physics/cij/change/{key}/{name}': value
                for key, row in result['error_changes'].items() for name, value in row.items()})
            metrics['tau/physics/completed_epochs'] = epoch
            for logger in trainer.loggers:
                logger.log_metrics(metrics, step=trainer.global_step)
        # Only rank0 retains the bootstrap baseline; all ranks share lifecycle.
        self.last_epoch = epoch
        dist.barrier()
