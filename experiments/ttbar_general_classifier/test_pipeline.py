"""Scientific invariants and native EveNet integration; all CPU, synthetic data."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.ttbar_general_classifier.core import *
from experiments.ttbar_general_classifier.run import (
    tensor_batch,sharded_batches,observed_condition,extract_features,normalization,
    write_json,CachedRows,load_policy,resolve_yaml,read_config,worker_device,
)


def extras(n):
    a={}
    for i in (1,2):
        for obj in ('b','l'):
            prefix=f'EXTRA/t{i}/{obj}'
            for v,val in [('pt',40.),('eta',.2*i),('phi',.7*i),('energy',60.)]:a[f'{prefix}/{v}']=np.full(n,val)
        for obj in ('t','l'):
            prefix=f'EXTRA/truth_t{i}/{obj}'
            for v,val in [('pt',100.),('eta',.1*i),('phi',-.5*i),('mass',173. if obj=='t' else .1)]:a[f'{prefix}/{v}']=np.full(n,val)
    return a


def save_part(root,name,a,role):
    d=root/name;d.mkdir();fields={}
    for j,(k,v) in enumerate(a.items()):np.save(d/f'{j}.npy',v);fields[k]=f'{j}.npy'
    write_json(d/'part.json',dict(role=role,events=len(a['truth']),fields=fields))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_interactive_ray_launch(self):
        from unittest.mock import patch,Mock
        from experiments.ttbar_general_classifier.run import ray_launch
        c=dict(evenet_root='/test/evenet',workers=4,cpus_per_worker=8)
        with patch('experiments.ttbar_general_classifier.run.eve_imports'), \
             patch('ray.init') as init, patch('ray.shutdown') as shutdown, \
             patch('ray.train.torch.TorchTrainer') as trainer, \
             patch('torch.cuda.device_count',return_value=4), \
             patch.dict(os.environ,{},clear=True):
            fn=Mock()
            ray_launch(fn,c)
            self.assertEqual(init.call_args.kwargs['address'],'local')
            self.assertEqual(trainer.call_args.kwargs['scaling_config'].num_workers,4)
            trainer.return_value.fit.assert_called_once()
            shutdown.assert_called_once()
            with patch.dict(os.environ,{'RAY_ADDRESS':'explicit-cluster'}):
                ray_launch(fn,c)
                self.assertEqual(init.call_args.kwargs['address'],'explicit-cluster')
            init.reset_mock()
            with patch('torch.cuda.device_count',return_value=0):
                with self.assertRaisesRegex(RuntimeError,'visible GPUs'):ray_launch(fn,c)
            init.assert_not_called()
            with patch.dict(os.environ,{'SLURM_JOB_NUM_NODES':'2'}):
                with self.assertRaisesRegex(ValueError,'Multi-node'):ray_launch(fn,c)
            trainer.return_value.fit.side_effect=RuntimeError('worker failed')
            shutdown.reset_mock()
            with self.assertRaisesRegex(RuntimeError,'worker failed'):ray_launch(fn,c)
            shutdown.assert_called_once()

    def test_ray_assigned_gpu_routing(self):
        from unittest.mock import patch
        with patch('ray.train.torch.get_device',return_value=torch.device('cuda:3')), patch('torch.cuda.set_device') as selected:
            self.assertEqual(worker_device(),torch.device('cuda:3'));selected.assert_called_once_with(torch.device('cuda:3'))

    def test_exact_general_head_operations(self):
        # Load only reference class definitions, avoiding unrelated tau CLI imports.
        namespace=dict(nn=nn,torch=torch,F=F,math=math,SPIN_DIM=15)
        for path,names in [(ROOT/'scripts/train_conditional_spin_ratio.py',['ConditionalSpinMLP']),
                           (ROOT/'scripts/tau_conditioning_heads.py',['ConditioningBlock','ConditionedTauMLP'])]:
            tree=ast.parse(path.read_text())
            module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
            exec(compile(module,str(path),'exec'),namespace)
        ref=namespace['ConditionedTauMLP'](9,hidden=256,dropout=0.,candidate_dim=23,kind='film',depth=3,
            condition_hidden=256,condition_width=256,process_label_index=7,num_process_classes=1)
        new=GeneralClassifier(7,23,dropout=0.,ratio_bound=None)
        ref.load_state_dict(new.state_dict(),strict=True)
        x=torch.randn(8,7);f=torch.randn(8,23);labels=torch.zeros(8,dtype=torch.long)
        cond=torch.cat((x,labels[:,None],torch.ones(8,1)),1)
        torch.testing.assert_close(ref(cond,f),new(x,f,labels))
        loss=F.softplus(-new(x,f,labels)).mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()))

    def test_top_oracle_mixture_and_p4(self):
        n=8;a=extras(n);x=np.zeros((n,3,2,3));x[...,0]=np.log1p(20.)
        p,t=top_oracle_inputs(x,a)
        visible=sum(np.stack([p4_from_columns(a,f'EXTRA/t{i}/{obj}') for i in (1,2)],1) for obj in ('b','l'))
        np.testing.assert_allclose(p[:,0]-visible,nu_p4(x[:,0]),atol=1e-12)
        # Set one candidate to truth exactly; other two are displaced in z or energy.
        p[:,1]=t;p[:,0]=t+np.array([0,0,0,5]);p[:,2]=t+np.array([10,0,0,0])
        score=np.tile([1.,-1.,2.],(n,1));scale=[100,50,50,50]
        general,mix,d=mixture(score,p,t,scale)
        self.assertTrue((d.argmin(1)==1).all());np.testing.assert_allclose(mix.sum(1),1)
        np.testing.assert_allclose(mix,.7*general+.3*np.eye(3)[np.ones(n,dtype=int)])
        np.testing.assert_allclose(mixture(score,p,t,scale,alpha=0)[1],general)
        self.assertTrue((mixture(score,p,t,scale,alpha=1)[1][:,1]==1).all())
        # Shared xyz scale means rigidly rotating all momenta preserves ranking/distance.
        q,_=np.linalg.qr(np.random.default_rng(42).normal(size=(3,3)))
        pr=p.copy();tr=t.copy();pr[...,1:]=pr[...,1:]@q;tr[...,1:]=tr[...,1:]@q
        np.testing.assert_allclose(mixture(score,pr,tr,scale)[2],d,atol=1e-12)

    def test_reco_truthnu_oracle_uses_same_visible_and_no_mc_top(self):
        rng=np.random.default_rng(42);n=5;k=8;a=extras(n)
        truth=rng.normal(size=(n,2,3));truth[...,0]=3.+abs(truth[...,0])
        candidates=np.repeat(truth[:,None],k,1)+rng.normal(scale=.1,size=(n,k,2,3))
        candidates[:,3]=truth
        p,t=top_oracle_inputs(candidates,a,truth,'reco_top_truthnu_p4')
        np.testing.assert_allclose(p-t[:,None],nu_p4(candidates)-nu_p4(truth)[:,None],atol=1e-12)
        scale=[100.,50.,50.,50.];logits=rng.normal(size=(n,k))
        _,w,d=mixture(logits,p,t,scale)
        self.assertTrue((d.argmin(1)==3).all());np.testing.assert_allclose(w.sum(1),1.)
        # Changing independent MC top fields cannot affect this oracle.
        poison={key:(value*123 if '/truth_t' in key else value.copy()) for key,value in a.items()}
        p2,t2=top_oracle_inputs(candidates,poison,truth,'reco_top_truthnu_p4')
        np.testing.assert_array_equal(t,t2);np.testing.assert_array_equal(p,p2)
        # Same fixed normalization: any shared visible shift cancels exactly.
        changed={key:(value+20 if '/energy' in key else value.copy()) for key,value in a.items()}
        p2,t2=top_oracle_inputs(candidates,changed,truth,'reco_top_truthnu_p4')
        np.testing.assert_allclose(mixture(logits,p2,t2,scale)[2],d,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'MC neutrino'):top_oracle_inputs(candidates,a,reference='reco_top_truthnu_p4')

    def test_event_covariance_and_native_bins(self):
        import pandas as pd
        for edges in NU2FLOW_EDGES.values():
            x=np.array([v for v in edges[1:-1] if np.isfinite(v)]+[.1,.7])
            if edges[0]!=-np.inf:x=x[x>edges[0]]
            if edges[-1]!=np.inf:x=x[x<=edges[-1]]
            np.testing.assert_array_equal(bin_ids(x,edges),pd.cut(x,edges,labels=False))
        h=contribution(np.array([[0,1],[1,1]]),np.full((2,2),.5),2)
        self.assertAlmostEqual((h.T@h)[0,1],.25)  # two samples in one event are correlated
        self.assertAlmostEqual((h.T@h).sum(),2.)

    def test_full_file_sharding_and_identity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for rowgroup in (9,1000):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'data.parquet';pq.write_table(pa.table({'i':np.arange(101)}),path,row_group_size=rowgroup)
                got=[];active=set()
                for rank in range(4):
                    for offset,b in sharded_batches(pq.ParquetFile(path),4,['i'],rank,4,0):
                        values=b.column(0).to_numpy();np.testing.assert_array_equal(values,np.arange(offset,offset+len(values)))
                        got.extend(values);active.add(rank)
                self.assertEqual(sorted(got),list(range(101)));self.assertEqual(active,set(range(4)))
        ids=np.array([f'val:{i}' for i in range(1000)])
        p=event_partition(ids);np.testing.assert_array_equal(event_partition(ids[::-1]),p[::-1]);self.assertEqual(set(p),set(range(4)))

    def test_ram_cache_preserves_shards_selection_padding_and_has_no_batch_io(self):
        from unittest.mock import patch
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);names=[];expected={k:[] for k in CachedRows.fields};selected={k:[] for k in CachedRows.fields}
            for shard,n in enumerate((5,3,7)):
                a=dict(truth=np.zeros((n,2,3)),condition=rng.normal(size=(n,7)).astype('float32'),
                    truth_features=rng.normal(size=(n,23)).astype('float32'),
                    generated_features=rng.normal(size=(n,8,23)).astype('float32'),
                    labels=np.zeros(n,dtype='int64'),partition=np.arange(n)%3)
                if shard==1:a['partition'][:]=1  # a shard with no selection rows
                name=f'part-{shard}';save_part(out,name,a,'validation');names.append(name)
                for key in expected:
                    v=a[key][:,0] if key=='generated_features' else a[key]
                    expected[key].append(v);selected[key].append(v[a['partition']==0])
            write_json(out/'CACHE_COMPLETE.json',dict(parts=names))
            full=CachedRows(out,'validation');sel=CachedRows(out,'validation',True)
            self.assertEqual(full.n,15)
            for rows,values in ((full,expected),(sel,selected)):
                values={k:np.concatenate(v) for k,v in values.items()}
                indices=np.array([rows.n-1,0,3,0,rows.n-1,1])  # shuffled + repeated padding
                with patch('numpy.load',side_effect=AssertionError('batch reopened a cache file')):
                    actual=rows.batch(indices)
                    for key in values:
                        np.testing.assert_array_equal(actual[key],values[key][indices])
                        self.assertEqual(actual[key].dtype,values[key].dtype)
                self.assertEqual(rows.nbytes,sum(v.nbytes for v in values.values()))
                self.assertTrue(all(v.flags.c_contiguous for v in rows.arrays.values()))

    def test_train_only_normalization_cache_roundtrip(self):
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name,role in [('train','train'),('val','validation')]:
                n=20;a=dict(truth=np.zeros((n,2,3)),candidates=np.zeros((n,8,2,3)),ids=np.array([f'{name}:{i}' for i in range(n)]),
                    condition=rng.normal(size=(n,7)).astype('float32'),truth_features=rng.normal(size=(n,23)).astype('float32'),
                    generated_features=rng.normal(size=(n,8,23)).astype('float32'),labels=np.zeros(n,dtype='int64'),
                    partition=np.arange(n)%4,oracle_valid=np.ones(n,dtype=bool),**extras(n))
                validate_cache_part(a);save_part(out,name,a,role)
            write_json(out/'CACHE_COMPLETE.json',dict(parts=['train','val']))
            st=normalization(dict(output=tmp));rows=CachedRows(tmp,'train');v=rows.batch(np.array([19,0,2]))
            np.testing.assert_allclose(v['generated_features'],np.load(out/'train/5.npy')[[19,0,2],0])
            for path in (out/'val').glob('*.npy'):
                v=np.load(path)
                if np.issubdtype(v.dtype,np.floating):np.save(path,v*999)
            st2=normalization(dict(output=tmp))
            torch.testing.assert_close(st['mean'],st2['mean']);self.assertEqual(st['oracle_scale'],st2['oracle_scale'])
            rc=dict(output=tmp,mixture=dict(oracle='reco_top_truthnu_p4'))
            new=normalization(rc)
            from experiments.ttbar_general_classifier.run import parts
            train=next(parts(tmp,'train'))[2];ref=oracle_reference_p4(train['truth'],train,'reco_top_truthnu_p4')
            ms=(ref**2).mean((0,1));np.testing.assert_allclose(new['oracle_scale'],[np.sqrt(ms[0])]+[np.sqrt(ms[1:].mean())]*3)
            self.assertEqual(new['oracle_reference'],'reco_top_truthnu_p4')

    def test_native_evenet_clean_features_and_sampler(self):
        eve=Path(os.environ.get('EVENET_ROOT','/Users/yirenwu/EveNet-private'))
        if not eve.exists():self.skipTest('Set EVENET_ROOT for native integration')
        sys.path.insert(0,str(eve))
        spec=importlib.util.spec_from_file_location('ttbar_fixture',eve/'tests/test_ttbar_visible_film.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        c,norm=fixture.config_and_norm()
        from evenet.network.evenet_model import build_evenet_model_from_training_config
        from evenet.network.loss.famo import FAMO
        from RL.DGPO_neutrino.dgpo_trainer import generate_neutrino_candidates
        from evenet.utilities.diffusion_sampler import DDIMSampler
        model=build_evenet_model_from_training_config(c,norm,torch.device('cpu')).eval().requires_grad_(False)
        a={k:v.numpy() for k,v in fixture.batch().items()};a['classification']=np.zeros(2);a['num_sequential_vectors']=np.array([4,4])
        b=tensor_batch(a,'cpu');self.assertEqual(b['classification'].dtype,torch.long)
        co=observed_condition(model,b);features=extract_features(model,b,b['x_invisible'])
        changed={k:v.clone() for k,v in b.items()};changed['x_invisible']+=100
        np.testing.assert_array_equal(co,observed_condition(model,changed))
        np.testing.assert_array_equal(features,extract_features(model,changed,b['x_invisible']))
        with torch.no_grad():
            torch.manual_seed(42);cand=generate_neutrino_candidates(model,b,DDIMSampler(torch.device('cpu')),K=2,num_ddim_steps=3,device=torch.device('cpu'))
            torch.manual_seed(42);cand2=generate_neutrino_candidates(model,changed,DDIMSampler(torch.device('cpu')),K=2,num_ddim_steps=3,device=torch.device('cpu'))
        torch.testing.assert_close(cand,cand2);self.assertEqual(cand.shape,(2,2,2,3));self.assertTrue(torch.isfinite(cand).all())
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        # Strict checkpoint loading including supervised Engine's dormant FAMO state.
        model.register_module('famo',FAMO(['classification','regression','assignment','generation','segmentation'],torch.device('cpu'),turn_on=False))
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            out=Path(tmp);torch.save(norm,out/'norm.pt');raw=resolve_yaml(eve/'reasoning_module/train_TT2L_visible_film_full.yaml');raw['network']=c.network.to_dict();raw['options']=c.options.to_dict();raw['options']['Dataset']['normalization_file']=str(out/'norm.pt')
            (out/'policy.yaml').write_text(yaml.safe_dump(raw,sort_keys=False));torch.save({'state_dict':{'model.'+k:v for k,v in model.state_dict().items()}},out/'policy.ckpt')
            restored=load_policy(dict(evenet_root=str(eve),output=tmp),torch.device('cpu'))
            np.testing.assert_allclose(features,extract_features(restored,b,b['x_invisible']),rtol=1e-5,atol=1e-6)

    def test_cached_evaluation_report_end_to_end(self):
        from unittest.mock import patch
        from experiments.ttbar_general_classifier.run import evaluate
        import json
        rng=np.random.default_rng(42);n=768;k=8
        truth=np.zeros((n,2,3));truth[...,0]=np.log1p(10+(np.arange(n)%4))[:,None]
        candidates=np.repeat(truth[:,None],k,1)
        bins=np.broadcast_to(np.arange(n)[:,None]%4,(n,k)).copy()
        migrate=rng.random((n,k))<.2;bins[migrate]=rng.integers(0,4,migrate.sum())
        candidates[...,0]=np.log1p(10+bins)[...,None]
        a=dict(ids=np.array([f'val:{i}' for i in range(n)]),truth=truth,candidates=candidates,
            partition=event_partition([f'val:{i}' for i in range(n)]),physics_keep=np.ones(n,dtype=bool),
            condition=rng.normal(size=(n,7)).astype('float32'),labels=np.zeros(n,dtype='int64'),
            truth_features=geometry(truth),generated_features=geometry(candidates),**extras(n))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);c=read_config(ROOT/'config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5.yaml');c['output']=tmp;c['evaluation']['response_bootstraps']=8
            save_part(out,'val',a,'validation');write_json(out/'CACHE_COMPLETE.json',dict(parts=['val']))
            write_json(out/'FIT_COMPLETE.json',dict(plateau=True));write_json(out/'cache_manifest.json',dict(source_checkpoint='SYNTHETIC TEST ONLY'))
            st=dict(condition_dim=7,candidate_dim=23,mean=torch.zeros(23),scale=torch.ones(23),oracle_scale=[100.,50.,50.,50.],oracle_reference='reco_top_truthnu_p4')
            model=GeneralClassifier(7,23,**c['head']);torch.save(dict(stats=st,state_dict=model.state_dict()),out/'classifier-best.pt')
            def targets(t,g,e,**kwargs):
                return {'synthetic':(np.rint(np.expm1(t[:,0,0])-10),np.rint(np.expm1(g[:,:,0,0])-10),np.arange(5)-.5,3)}
            with patch('experiments.ttbar_general_classifier.run.physics',targets):evaluate(c)
            r=json.loads((out/'evaluation/report.json').read_text());self.assertEqual(r['targets']['synthetic']['status'],'complete')
            for name in ('distribution','uncertainty','response','unfolded'):self.assertTrue((out/f'evaluation/synthetic/{name}.png').exists())
            self.assertTrue((out/'evaluation/index.html').is_file())

    def test_actual_roounfold_identity_closure(self):
        from scripts.tau_candidate_svd_evaluation import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            truth=np.repeat(np.arange(4),100);h=np.eye(4)[truth]
            result=evaluate([h,h],[h,h],truth,truth,truth,np.arange(5),Path(tmp),names=['Raw','General'],kreg=4,reps=8)
            for m in result['models']:
                self.assertLess(m['closure']['max_abs_pull'],1e-5)
                self.assertGreater(m['total_shape_rss'],0)
            self.assertAlmostEqual(result['comparisons']['General / Raw']['total_ratio'],1.)

if __name__=='__main__':unittest.main()
