"""Verify frozen evidence or replay independent computations in a clean copy."""
from pathlib import Path
import argparse,hashlib,json,statistics,shutil,subprocess,sys,uuid,datetime
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');p.add_argument('--work-dir');a=p.parse_args()
 load=lambda name:json.loads((ROOT/name).read_text())
 manifest=load('MANIFEST_SHA256.json')
 for n,h in manifest.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 rows=load('results/nonlinear_test_rows.json');counts={}
 for target in sorted(set(r['target'] for r in rows)):
  rr=[r for r in rows if r['target']==target]
  counts[target]=[sum((r[k][1]-r[k][0])/2+1e-5<=r['tolerance'] for r in rr) for k in ['box_interval','window_interval','interval']]
 assert counts=={'front_mean_N':[162,162,162],'front_change_N':[31,84,222],'axle_contrast_N':[38,379,379],'total_impulse_Ns':[26,53,53]}
 assert len(rows)==1728 and all(r['included'] for r in rows)
 prior=load('results/final_paired_priors.json')['rows']
 pc=[[sum(r['report'] for r in prior if r['setting']==s and r['method']==m) for m in ['box','window','shared']] for s in ['known_vehicle','wide_centered','wide_offcenter']]
 assert pc==[[5,20,46],[3,16,28],[3,16,29]]
 ratios=[]
 for r in load('results/v7_independent_benchmark.json')['rows']:
  d=r['measurements']['raw_dense'];c=r['measurements']['compressed'];ratios.append((d['setup']['median']+d['queries']['16']['median'])/(c['setup']['median']+c['queries']['16']['median']))
 assert len(ratios)==15 and round(statistics.median(ratios),2)==4.63
 state=dict(state='PASS',run_id=str(uuid.uuid4()),hashes=len(manifest),counts=counts,paired_prior_counts=pc,archived_timing_ratio_median=statistics.median(ratios),fresh_inference=False)
 if a.replay:
  if not a.work_dir:raise ValueError('--replay requires --work-dir')
  w=Path(a.work_dir).resolve()
  if w==ROOT or ROOT in w.parents:raise ValueError('Use a new directory outside the package')
  w.mkdir(parents=True,exist_ok=False);state['state']='RUNNING';(w/'RUN_STATE.json').write_text(json.dumps(state,indent=2));project=w/'project'
  try:
   shutil.copytree(ROOT,project,ignore=shutil.ignore_patterns('.git','__pycache__','*.nbc','*.nbi'))
   for i,cmd in enumerate([['code/verify_v7.py','verify'],['code/public_dense_replay.py']]):
    with (w/f'{i+1:02d}_replay.log').open('w') as log:subprocess.run([sys.executable,*cmd],cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
   v=json.loads((project/'results/v7_dense_verification.json').read_text());r=json.loads((project/'results/public_dense_replay.json').read_text())
   assert v['passed'] and v['checks']==144 and r['passed'] and r['queries']==1728 and r['decision_changes']==0
   state.update(state='PASS',fresh_inference=True,sensor_checks=v['checks'],replayed_queries=r['queries'],max_interval_gap=r['max_interval_gap'],decision_changes=r['decision_changes'])
  except Exception as ex:state.update(state='FAILED',error=str(ex));raise
  finally:(w/'RUN_STATE.json').write_text(json.dumps(state,indent=2))
 print(json.dumps(state,indent=2))
if __name__=='__main__':main()
