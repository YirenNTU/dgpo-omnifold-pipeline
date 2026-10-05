"""No-GPU checks for the YAML entrypoint and 10% DGPO data contract."""
from __future__ import annotations

import tempfile
import unittest
import ast
import argparse
from pathlib import Path

import yaml

from scripts.run_conditional_tau_ratio import command, read_config


CONFIG = Path(__file__).resolve().parents[1] / 'config/conditional_tau_ratio_10pct.yaml'


class ConditionalTauYamlTest(unittest.TestCase):
    def test_worker_config_contract_and_min_delta_cli(self):
        # Inspect the actual parser/config without importing the GPU dependency stack.
        source = CONFIG.parents[1] / 'scripts/train_conditional_spin_ratio.py'
        tree = ast.parse(source.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training_worker')
        required = {n.slice.value for n in ast.walk(worker) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == 'cfg'
            and isinstance(n.slice, ast.Constant)}
        assignment = next(n for n in main.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'cfg' for t in n.targets))
        supplied = {kw.arg for kw in assignment.value.keywords}
        self.assertEqual(required - supplied, set(), 'Worker config keys missing from driver')
        min_delta = next(kw.value for kw in assignment.value.keywords if kw.arg == 'min_delta')
        self.assertEqual(ast.unparse(min_delta), 'args.min_delta')
        cli = next(n for n in main.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and n.value.args
            and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == '--min-delta')
        parser = argparse.ArgumentParser()
        exec(compile(ast.Module(body=[cli], type_ignores=[]), str(source), 'exec'), {'parser':parser})
        cfg = read_config(CONFIG)
        cfg['classifier']['min_delta'] = 0.001
        cmd = command(cfg, 'train')
        parsed, _ = parser.parse_known_args(cmd[2:])
        self.assertEqual(parsed.min_delta, 0.001)

    def test_10pct_data_paths_match_dgpo(self):
        cfg = read_config(CONFIG)
        self.assertIn('omnifold_attention_10pct_stic_filtered_test1/train',
            cfg['platform']['data_parquet_dir'])
        self.assertIn('diffusion_val_20pct_seed42_stic_filtered_test1/val',
            cfg['platform']['data_parquet_val_dir'])
        self.assertEqual(cfg['sampling']['workers'], 16)
        self.assertEqual(cfg['classifier']['workers'], 16)

    def test_prepare_reuses_saved_samples_without_ray(self):
        cfg = read_config(CONFIG)
        cmd = command(cfg, 'prepare', 'example:6379')
        self.assertIn('--prepare-only', cmd)
        self.assertNotIn('--ray-address', cmd)
        self.assertEqual(cmd[cmd.index('--train-source') + 1],
            cfg['experiment']['train_sample_root'])
        self.assertEqual(cmd[cmd.index('--train-events') + 1],
            cfg['platform']['data_parquet_dir'])
        self.assertEqual(cmd[cmd.index('--test-events') + 1],
            cfg['platform']['data_parquet_val_dir'])

    def test_train_uses_same_data_and_16_workers(self):
        cfg = read_config(CONFIG)
        cmd = command(cfg, 'train', 'example:6379')
        self.assertEqual(cmd[cmd.index('--workers') + 1], '16')
        self.assertEqual(cmd[cmd.index('--representation') + 1], 'tau')
        self.assertEqual(cmd[cmd.index('--ray-address') + 1], 'example:6379')
        self.assertEqual(cmd[cmd.index('--run-name') + 1],
            cfg['logger']['wandb']['classifier_run_name'])

    def test_mismatched_validation_pool_fails_before_launch(self):
        cfg = read_config(CONFIG)
        cfg['platform']['data_parquet_val_dir'] = '/wrong/validation'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, 'differs from 10% DGPO'):
                read_config(path)


if __name__ == '__main__':
    unittest.main()
