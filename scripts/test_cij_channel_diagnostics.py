import unittest
import numpy as np
from scripts.cij_channel_diagnostics import channel_diagnostics


class ChannelTests(unittest.TestCase):
    def fixture(self):
        a=np.tile([1.,0.,0.],(8,1))
        b=a.copy(); b[4:]*=-1
        return dict(event_id=np.arange(8),truth_a=a,truth_b=b,sample_a=a[:,None],
            sample_b=b[:,None],log_ratio=np.log(np.array([1.]*4+[3.]*4))[:,None],
            event_weight=np.ones(8)),np.array([11]*4+[44]*4)

    def test_mixture_only(self):
        data,category=self.fixture()
        report=channel_diagnostics(data,[1,1],category,'joint',20)
        d=report['decomposition']
        np.testing.assert_allclose(d['within_channel_shift'],0,atol=1e-12)
        self.assertAlmostEqual(d['mixture_shift'][0][0],-4.5)
        self.assertLess(d['reconstruction_error'],1e-12)
        self.assertAlmostEqual(report['channels'][0]['reweighted_fraction'],.25)

    def test_within_only(self):
        data,category=self.fixture()
        data['log_ratio'][:]=0
        data['sample_b'][0,0]*=-1
        report=channel_diagnostics(data,np.ones((8,2)),category,'joint',20,
            references={'matched_target':(data['truth_a'],data['truth_b'])})
        np.testing.assert_allclose(report['decomposition']['mixture_shift'],0,atol=1e-12)
        self.assertIn('matched_target',report['decomposition']['errors'])

    def test_unsupported_channel(self):
        data,category=self.fixture()
        data['log_ratio'][4:]=-1000
        report=channel_diagnostics(data,[1,1],category,'joint',20)
        self.assertFalse(report['decomposition']['available'])

    def test_direct_moments_ignore_kappa_and_factor_nine(self):
        data,category=self.fixture()
        # Nontrivial within-channel reweighting with known direct moment.
        data['sample_b']=data['sample_b'].copy()
        data['sample_b'][0,0]*=-1
        data['log_ratio'][0,0]=np.log(3.)
        refs={'matched_target':(data['truth_a'],data['truth_b'])}
        report=channel_diagnostics(data,[-.33,.41],category,'joint',30,references=refs)
        r=report['channels'][0]['angular_moments']['references']['matched_target']
        self.assertAlmostEqual(r['results']['truth']['moment'][0][0],1.)
        self.assertAlmostEqual(r['results']['unweighted']['moment'][0][0],.5)
        self.assertAlmostEqual(r['results']['reweighted']['moment'][0][0],0.)
        self.assertAlmostEqual(r['absolute_error_change'][0][0],.5)
        self.assertAlmostEqual(r['error_change']['value'],.5)
        other=channel_diagnostics(data,[1,1],category,'joint',30,references=refs)
        self.assertEqual(report['channels'][0]['angular_moments'],other['channels'][0]['angular_moments'])


if __name__=='__main__': unittest.main()
