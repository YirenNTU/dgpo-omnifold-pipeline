"""Replication uncertainty must never be silently replaced by event uncertainty."""
import copy
import ast
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evenet_dgpo'))
from RL.DGPO_neutrino.tau_direction_repeat_statistics import summarize_direction_repeats


def fixture():
    draws = {}
    for draw in range(4):
        rows = {}
        for direction, slope in (('raw_reward', 2.), ('raw_total', 1.), ('native_adamw', -.5)):
            for fraction in (.01, .03, .1):
                for sign in (-1, 1):
                    evaluations = {}
                    for seed, mc in (('101', -.001), ('102', .001)):
                        mean = sign * slope * fraction + .1 * fraction**2 + mc + draw*.0001
                        evaluations[seed] = {'reward': dict(delta_mean=mean,
                            delta_lo95=mean-.00001, delta_hi95=mean+.00001)}
                    rows[f'{direction}_{fraction}_{sign}'] = dict(direction=direction, fraction=fraction,
                        sign=sign, evaluations=evaluations)
        draws[str(draw)] = dict(interventions=rows)
    return draws


class DirectionStatisticsTests(unittest.TestCase):
    def test_read_only_wandb_metrics_use_draw_axis_not_policy_clock(self):
        source = Path(__file__).resolve().parents[1] / 'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'
        tree = ast.parse(source.read_text())
        axes = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_wandb_define_axes')
        calls = [node for node in ast.walk(axes) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'define_metric' and node.args and isinstance(node.args[0], ast.Constant)
                 and node.args[0].value == 'tau/local_direction/*']
        self.assertEqual(len(calls), 1)
        settings = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        self.assertEqual(settings['step_metric'], 'tau/local_direction/draw_index')
        self.assertFalse(settings['step_sync'])

    def test_draws_and_evaluation_noise_are_separate_and_serializable(self):
        draws = fixture(); before = copy.deepcopy(draws)
        report = summarize_direction_repeats(draws)
        self.assertEqual(draws, before)
        self.assertEqual(report['draw_count'], 4)
        self.assertEqual(report['evaluation_seed_count'], 2)
        cell = report['interventions']['raw_reward_rms0.01_sign+1']
        self.assertEqual(cell['update_draw_means']['count'], 4)
        self.assertEqual(cell['update_draw_means']['positive_count'], 4)
        self.assertAlmostEqual(cell['evaluation_noise_ranges']['mean'], .002)
        self.assertEqual(cell['all_conditional_ci_positive_draws'], 4)
        self.assertNotIn('lo95', cell['update_draw_means'])
        json.dumps(report, allow_nan=False)

    def test_signed_odd_even_and_central_slope(self):
        report = summarize_direction_repeats(fixture())
        response = report['signed_response']['raw_reward_rms0.1']
        self.assertAlmostEqual(response['odd_response']['mean'], .2)
        self.assertAlmostEqual(response['central_slope']['mean'], 2.)
        self.assertAlmostEqual(response['even_response']['mean'], .00115)

    def test_direction_contrast_is_descriptive_not_subtracted_confidence_interval(self):
        report = summarize_direction_repeats(fixture())
        contrast = report['direction_contrasts']['raw_total_minus_raw_reward_rms0.1_sign+1']
        self.assertAlmostEqual(contrast['across_draws']['mean'], -.1)
        self.assertFalse(contrast['paired_ci_available'])
        self.assertNotIn('lo95', contrast['across_draws'])

    def test_update_noise_can_reverse_sign_despite_tight_event_intervals(self):
        draws = fixture()
        key = 'raw_reward_0.01_1'
        for draw in ('1', '3'):
            for evaluation in draws[draw]['interventions'][key]['evaluations'].values():
                evaluation['reward'] = dict(delta_mean=-.1, delta_lo95=-.101, delta_hi95=-.099)
        cell = summarize_direction_repeats(draws)['interventions']['raw_reward_rms0.01_sign+1']
        self.assertEqual(cell['update_draw_means']['positive_count'], 2)
        self.assertEqual(cell['update_draw_means']['negative_count'], 2)
        self.assertEqual(cell['all_conditional_ci_negative_draws'], 2)

    def test_evaluation_mc_sign_reversal_is_not_hidden_by_mean(self):
        draws = fixture()
        evaluations = draws['0']['interventions']['raw_reward_0.01_1']['evaluations']
        evaluations['101']['reward'] = dict(delta_mean=-.01, delta_lo95=-.02, delta_hi95=0.)
        evaluations['102']['reward'] = dict(delta_mean=.03, delta_lo95=.02, delta_hi95=.04)
        cell = summarize_direction_repeats(draws)['interventions']['raw_reward_rms0.01_sign+1']
        self.assertGreater(cell['draws']['0']['evaluation_noise_mean'], 0.)
        self.assertFalse(cell['draws']['0']['all_evaluation_point_estimates_positive'])
        self.assertFalse(cell['draws']['0']['all_conditional_event_intervals_positive'])

    def test_partial_summary_and_zero_odd_are_honest(self):
        self.assertEqual(summarize_direction_repeats({})['draw_count'], 0)
        draws = fixture()
        for intervention in draws['0']['interventions'].values():
            for evaluation in intervention['evaluations'].values():
                evaluation['reward'] = dict(delta_mean=0., delta_lo95=-1., delta_hi95=1.)
        report = summarize_direction_repeats({'0': draws['0']})
        self.assertIsNone(report['signed_response']['raw_reward_rms0.01']['draws']['0']['absolute_even_to_odd'])
        self.assertIsNone(report['interventions']['raw_reward_rms0.01_sign+1']['update_draw_means']['sample_sd'])
        json.dumps(report, allow_nan=False)

    def test_unmatched_evaluation_seeds_rejected(self):
        draws = fixture()
        evaluation = draws['2']['interventions']['raw_total_0.1_-1']['evaluations']
        evaluation['103'] = evaluation.pop('102')
        with self.assertRaisesRegex(ValueError, 'seeds must match'):
            summarize_direction_repeats(draws)

    def test_unresolvable_perturbations_do_not_count_as_failed_directions(self):
        draws = fixture()
        for draw in draws.values():
            for sign in (-1, 1):
                for evaluated in draw['interventions'][f'raw_reward_0.01_{sign}']['evaluations'].values():
                    evaluated['measurement_valid'] = False
                    evaluated['reward'] = dict(delta_mean=0., delta_lo95=0., delta_hi95=0.)
        report = summarize_direction_repeats(draws)
        cell = report['interventions']['raw_reward_rms0.01_sign+1']
        self.assertEqual(cell['invalid_draw_count'], 4)
        self.assertEqual(cell['update_draw_means']['count'], 0)
        self.assertIsNone(cell['update_draw_means']['mean'])
        self.assertEqual(cell['raw_including_invalid_update_draw_means']['count'], 4)
        self.assertNotIn('raw_reward_rms0.01', report['signed_response'])
        self.assertNotIn('raw_total_minus_raw_reward_rms0.01_sign+1', report['direction_contrasts'])
        json.dumps(report, allow_nan=False)

    def test_nonfinite_invalid_radius_and_duplicate_cell_rejected(self):
        for mutation in ('nonfinite', 'radius', 'duplicate', 'sign', 'interval', 'seeds'):
            draws = fixture(); row = draws['0']['interventions']['raw_reward_0.01_1']
            if mutation == 'nonfinite': row['evaluations']['101']['reward']['delta_mean'] = float('nan')
            if mutation == 'radius': row['fraction'] = 0
            if mutation == 'sign': row['sign'] = True
            if mutation == 'interval': row['evaluations']['101']['reward']['delta_lo95'] = 100
            if mutation == 'seeds': row['evaluations'].pop('102')
            if mutation == 'duplicate': draws['0']['interventions']['duplicate'] = copy.deepcopy(row)
            with self.assertRaises(ValueError, msg=mutation): summarize_direction_repeats(draws)


if __name__ == '__main__':
    unittest.main()
