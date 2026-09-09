"""Execute the real batch shell/config preparer with fake Slurm/Shifter/Ray.

These are launch-contract tests, NOT a container, GPU or distributed smoke test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from train_neutrino_backend import REPO_ROOT, read_yaml


MOCK_COMMAND = r'''
import json, os, pathlib, subprocess, sys, time
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['MOCK_CALL_LOG'], 'a') as handle:
    handle.write(json.dumps({'name': name, 'args': args,
        'run_id': os.environ.get('WANDB_RUN_ID'),
        'ray_address': os.environ.get('RAY_ADDRESS')}) + '\n')
if name == 'scontrol':
    print('nid000001\nnid000002\nnid000003\nnid000004')
elif name == 'sleep':
    time.sleep(0.03)
elif name == 'srun':
    if 'hostname' in args:
        print('128.55.65.207 192.0.2.1')
    else:
        if '--head' in args:
            print('Ray runtime started.', flush=True)
        while True:
            time.sleep(0.1)
elif name == 'shifter':
    if 'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py' in args:
        # Import the production YAML loader; don't initialize a model or training.
        from evenet.control.global_config import Config
        cfg = Config()
        cfg.load_yaml(args[2])
        assert cfg.dgpo.checkpoint_load_mode == 'resume'
        assert cfg.platform.number_of_workers == 16
        assert cfg.dgpo.adaptive_omnifold.trigger.required_consecutive_checks == 8
        assert cfg.dgpo.validation_every_n_epochs == 10
        assert cfg.dgpo.validation_full_every_n_epochs == 10
        assert not cfg.dgpo.adaptive_omnifold.trigger.global_confirm_candidates
        assert cfg.dgpo.adaptive_omnifold.recalibration.policy_warmup_steps == 10
        assert cfg.dgpo.adaptive_omnifold.staleness_every_n_steps == 5
        assert cfg.dgpo.lr_schedule.type == 'cosine'
        print('MOCK_TRAINER_REACHED_WITH_VALID_CONFIG', flush=True)
    else:
        assert args[0] == 'python3', args
        raise SystemExit(subprocess.call([sys.executable, *args[1:]]))
else:
    raise SystemExit('Unexpected command ' + name)
'''

MOCK_RAY = '''
import os
def init(**kwargs):
    assert kwargs['address'] == '128.55.65.207:6379'
    if os.environ.get('MOCK_RAY_BAD'):
        raise SystemExit('MOCK Ray cluster unavailable')
def nodes():
    return [{'Alive': True, 'Resources': {'GPU': 4}} for _ in range(4)]
def cluster_resources():
    return {'GPU': 16}
def shutdown():
    pass
'''


class TestBatchLaunch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dgpo batch test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Supply a test-only event schema. Production's generated_event_info.yaml
        # is intentionally NERSC-local, and must NOT be regenerated on resume.
        self.repo = self.root / "repo"
        shutil.copytree(REPO_ROOT / 'config', self.repo / 'config')
        for name in ('scripts', 'evenet_dgpo'):
            (self.repo / name).symlink_to(REPO_ROOT / name, target_is_directory=True)
        subprocess.run([
            sys.executable, str(REPO_ROOT / 'generate_event_info_yaml.py'),
            '--analysis-config', str(REPO_ROOT / 'config/analysis.yaml'),
            '--evenet-config', str(REPO_ROOT / 'config/evenet_schema.yaml'),
            '--output', str(self.repo / 'config/generated_event_info.yaml'),
        ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        binary = self.root / "bin"
        binary.mkdir()
        for name in ("scontrol", "srun", "shifter", "sleep"):
            script = binary / name
            script.write_text(f"#!{sys.executable}\n" + MOCK_COMMAND)
            script.chmod(0o755)
        (self.root / "ray.py").write_text(MOCK_RAY)
        self.checkpoint = self.root / "source/checkpoints/snapshot.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"checkpoint fixture; only config loading is tested")
        self.last = self.checkpoint.parent / "last.ckpt"
        self.last.symlink_to(self.checkpoint.name)
        self.output = self.root / "new output"
        self.call_log = self.root / "calls.jsonl"
        self.env = dict(os.environ)
        self.env.update(
            PATH=str(binary) + os.pathsep + self.env["PATH"],
            PYTHONPATH=str(self.root),
            SLURM_SUBMIT_DIR=str(self.repo), SLURM_JOB_NUM_NODES="4",
            SLURM_JOB_ID="12345", SLURM_JOB_NODELIST="nid[000001-000004]",
            DGPO_RESUME_CHECKPOINT=str(self.last), DGPO_RUN_ROOT=str(self.output),
            MOCK_CALL_LOG=str(self.call_log), WANDB_RUN_ID="must-not-reuse",
        )
        self.env.pop("DGPO_REPO_ROOT", None)
        self.env.pop("MOCK_RAY_BAD", None)

    def launch(self):
        # Real sbatch scripts are spooled outside the repository too.
        spooled = self.root / "slurm_script"
        spooled.write_bytes((REPO_ROOT / "NERSC/submit-dgpo-10pct-resume.sbatch").read_bytes())
        result = subprocess.run(["bash", str(spooled)], cwd=self.root, env=self.env,
                                text=True, capture_output=True, timeout=30)
        calls = [json.loads(line) for line in self.call_log.read_text().splitlines()]
        return result, calls

    def test_success_real_shell_real_yaml_loader(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MOCK_TRAINER_REACHED_WITH_VALID_CONFIG", result.stdout)
        self.assertIn("Ray ready: 4 GPU nodes / 16 GPUs", result.stdout)
        cfg = read_yaml(self.output / "runtime.yaml")
        self.assertEqual(cfg["options"]["Training"]["model_checkpoint_load_path"], str(self.checkpoint))
        self.assertEqual(cfg["options"]["Training"]["model_checkpoint_save_path"], str(self.output / "checkpoints"))
        self.assertEqual(sorted(p.name for p in self.checkpoint.parent.iterdir()), ["last.ckpt", "snapshot.ckpt"])
        self.assertEqual(self.last.resolve(), self.checkpoint)
        self.assertTrue(all(call['run_id'] is None for call in calls))
        starts = [call for call in calls if call['name'] == 'srun' and 'start' in call['args']]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all('--gpus-per-task=4' in call['args'] for call in starts))

    def test_missing_checkpoint_never_starts_ray(self):
        self.env['DGPO_RESUME_CHECKPOINT'] = str(self.root / 'missing.ckpt')
        result, calls = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(any(call['name'] == 'srun' for call in calls))

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'runtime.yaml'
        sentinel.write_text('previous result: preserve me')
        result, calls = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), 'previous result: preserve me')
        self.assertFalse(any(call['name'] == 'srun' for call in calls))

    def test_failed_cluster_never_starts_training(self):
        self.env['MOCK_RAY_BAD'] = '1'
        result, calls = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MOCK Ray cluster unavailable', result.stderr)
        self.assertNotIn('MOCK_TRAINER_REACHED', result.stdout)
        self.assertFalse(any('evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py' in call['args'] for call in calls))

    def test_missing_event_schema_fails_before_ray(self):
        (self.repo / 'config/generated_event_info.yaml').unlink()
        result, calls = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('generated_event_info.yaml', result.stderr)
        self.assertFalse(any(call['name'] == 'srun' for call in calls))


if __name__ == '__main__':
    unittest.main()
