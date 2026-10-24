"""Manual CLI. --check performs no training, Ray launch, or allocation."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path);p.add_argument('--check',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--no-wandb',action='store_true')
    a=p.parse_args()
    import yaml
    c=yaml.safe_load(a.config.read_text())
    native_root=Path(c['evenet_root']);sys.path.insert(0,str(native_root))
    os.environ['PYTHONPATH']=os.pathsep.join((str(native_root),str(ROOT),os.environ.get('PYTHONPATH','')))
    from experiments.ttbar_dgpo_general.runtime import prepare,train_worker
    from experiments.ttbar_general_classifier.run import eve_imports
    eve_imports(c)
    runtime=prepare(c,a.resume)
    print(f'Prepared {runtime}: 16 GPUs x 512 events, K8, General70 + oracle30; refit every 500 steps',flush=True)
    if a.check:return
    import ray
    # A multi-node allocation/cluster is prepared by the user. Fail immediately;
    # never silently fall back to one local node or wait forever for missing GPUs.
    ray.init(address=os.environ.get('RAY_ADDRESS','auto'),runtime_env={'env_vars':{'PYTHONPATH':os.environ['PYTHONPATH']}},ignore_reinit_error=True)
    if ray.cluster_resources().get('GPU',0)<16:raise RuntimeError('Requires an existing 16-GPU Ray cluster')
    from RL.DGPO_neutrino import dgpo_trainer as native
    from experiments.ttbar_dgpo_general.preprocessing_bridge import bind_process_fn
    native.make_process_fn=bind_process_fn(native.make_process_fn,native_root)
    native.dgpo_train_loop=train_worker
    sys.argv=[sys.argv[0],str(runtime),'--max-steps',str(c['max_steps']),'--ray-dir',str(Path(c['output'])/'ray-results')]
    if a.no_wandb:sys.argv+=['--no-wandb']
    native.main()

if __name__=='__main__':main()
