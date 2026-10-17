"""Restricted-data-free, synthetic-only clean-directory reproduction."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,traceback,uuid
ROOT=Path(__file__).resolve().parent
SKIP={'wall_s','setup_s','query_s','elapsed_s','time_s','native_array_sha256','sensor_sha256'}
def compare(a,b,path='',stats=None):
    import math
    if stats is None:stats=dict(numbers=0,max_abs=0.,mismatches=[])
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):stats['mismatches'].append(path+':keys')
        for k in set(a)&set(b):
            if k not in SKIP:compare(a[k],b[k],path+'/'+k,stats)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):stats['mismatches'].append(path+':length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),stats)
    elif isinstance(a,bool) or isinstance(b,bool):
        if a!=b:stats['mismatches'].append(path+':decision')
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        stats['numbers']+=1;stats['max_abs']=max(stats['max_abs'],abs(a-b))
        if not math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-8):stats['mismatches'].append(path+':number')
    elif a!=b:stats['mismatches'].append(path+':value')
    return stats
def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True);p.add_argument('--inject-failure',action='store_true');p.add_argument('--inject-missing-dependency',action='store_true');a=p.parse_args()
    work=Path(a.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    state=dict(run_id=str(uuid.uuid4()),state='RUNNING',passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=[],python=sys.version)
    def write():(work/'RUN_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    write()
    try:
        if a.inject_failure:raise RuntimeError('Deliberate public failure regression')
        if a.inject_missing_dependency:__import__('mssp_missing_public_dependency_regression')
        if work==ROOT or ROOT in work.parents:raise ValueError('Work directory must be outside package')
        project=work/'project'
        if project.exists():raise ValueError('Fresh work directory required')
        for name,digest in json.loads((ROOT/'MANIFEST_SHA256.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        references=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['deterministic_reference_results']
        refs={n:json.loads((ROOT/'results'/n).read_text()) for n in references};budget=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text())
        budget_bytes=(ROOT/'data/nonlinear_frozen_budget.json').read_bytes()
        shutil.copytree(ROOT,project,ignore=shutil.ignore_patterns('__pycache__','*.nbc','*.nbi'))
        commands=[['code/experiments_v6.py','development'],['code/experiments_v6.py','test'],['code/verify_v7.py','verify'],['code/check_input_contract_v6.py'],['code/public_dense_replay.py'],['code/sensitivity_v7.py'],['code/generic_force_v7.py'],['code/final_paired_priors.py'],['code/final_frontier.py']]
        for i,cmd in enumerate(commands):
            log=work/f'{i+1:02d}_{Path(cmd[0]).stem}.log'
            with log.open('w') as stream:q=subprocess.run([sys.executable,*cmd],cwd=project,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'MPLBACKEND':'Agg'})
            state['commands'].append(dict(command=cmd,returncode=q.returncode));write()
            if q.returncode:raise RuntimeError('Stage failed: '+str(cmd))
            if cmd==['code/experiments_v6.py','development']:
                path=project/'data/nonlinear_frozen_budget.json';checked=compare(budget,json.loads(path.read_text()))
                if checked['mismatches']:raise AssertionError('Independent recalibration differs')
                shutil.copy2(path,path.with_name('nonlinear_budget_reexecuted.json'));path.write_bytes(budget_bytes)
                lock=project/'reproducibility/frozen_budget_manifest.json';info=json.loads(lock.read_text());info['budget_sha256']=hashlib.sha256(budget_bytes).hexdigest();lock.write_text(json.dumps(info,indent=2)+'\n')
                state['development_recalibration']=dict(**checked,published_budget_restored_byte_for_byte=True);write()
        comparisons={n:compare(ref,json.loads((project/'results'/n).read_text())) for n,ref in refs.items()}
        comparisons['frozen_budget']=compare(budget,json.loads((project/'data/nonlinear_frozen_budget.json').read_text()))
        assert not any(v['mismatches'] for v in comparisons.values()),comparisons
        state.update(state='PASS',passed=True,comparisons=comparisons,scope='Synthetic-only numerical reproduction, 24/48 split and 8 reused prior trajectories. No CarSim, manuscript PDF or fresh runtime benchmarking in this public scope; archived benchmark inputs/results included.')
    except BaseException as e:state.update(state='FAILED',passed=False,error=str(e),traceback=traceback.format_exc());write();raise
    finally:state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write()
    print(json.dumps(dict(state=state['state'],run_id=state['run_id'])))
if __name__=='__main__':main()
