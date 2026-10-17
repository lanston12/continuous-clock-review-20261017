"""Threshold enumeration on frozen intervals, retaining run-level paired units."""
from pathlib import Path
import json,hashlib
import numpy as np
from experiments_v6 import dump
ROOT=Path(__file__).resolve().parents[1]
KEYS={'box':'box_interval','window':'window_interval','shared':'interval'}
def main():
    P=json.loads((ROOT/'reproducibility/protocol_final_strengthening.json').read_text())
    primary=[r for r in json.loads((ROOT/'results/nonlinear_test_rows.json').read_text()) if r['target']=='front_change_N']
    paired=json.loads((ROOT/'results/final_paired_priors.json').read_text())
    normalized=[]
    for r in primary:
        for method,key in KEYS.items():
            v=r[key];normalized.append(dict(population='primary_48',setting='known_vehicle',record=r['record'],mode=r['maneuver'],tire_law=r['tire_law'],method=method,radius_N=(v[1]-v[0])/2,full_class_pass=r['reference_class_pass']))
    for r in paired['rows']:
        normalized.append(dict(population='sensitivity_8',setting=r['setting'],record=r['record'],mode=r['mode'],tire_law=r['tire_law'],method=r['method'],radius_N=r['radius_N'],full_class_pass=r['full_class_pass']))
    curves=[];runrows=[]
    for population,setting in sorted(set((r['population'],r['setting']) for r in normalized)):
        pop=[r for r in normalized if r['population']==population and r['setting']==setting]
        strata=[('all',None)]+[('mode',i) for i in range(4)]+[('tire_law',i) for i in range(2)]
        for field,value in strata:
            sub=pop if field=='all' else [r for r in pop if r[field]==value]
            for tau in P['tolerance_grid_N']:
                for method in KEYS:
                    rr=[r for r in sub if r['method']==method];n=len(rr);count=sum(r['radius_N']+P['decision_guard_N']<=tau for r in rr)
                    curves.append(dict(population=population,setting=setting,stratum=field,stratum_value=value,method=method,tolerance_N=tau,reports=count,queries=n,independent_runs=len(set(r['record'] for r in rr)),fraction=count/n,class_valid_runs=len(set(r['record'] for r in rr if r['full_class_pass']))))
        for record in sorted(set(r['record'] for r in pop)):
            rr=[r for r in pop if r['record']==record]
            rads={k:[r['radius_N'] for r in rr if r['method']==k] for k in KEYS}
            runrows.append(dict(population=population,setting=setting,record=record,mode=rr[0]['mode'],tire_law=rr[0]['tire_law'],
                mean_radius_N={k:float(np.mean(v)) for k,v in rads.items()},width_reduction=1-float(np.mean(rads['shared'])/np.mean(rads['window'])),
                report_gain_pp={str(tau):100*(sum(x+P['decision_guard_N']<=tau for x in rads['shared'])-sum(x+P['decision_guard_N']<=tau for x in rads['window']))/9 for tau in P['tolerance_grid_N']}))
    scales=[]
    for record in sorted(set(r['record'] for r in primary)):
        m,I,a,b=np.load(ROOT/'data/nonlinear_test'/(record+'.npz'))['mechanics']
        scale=m*9.80665*b/(a+b)
        scales.append(dict(record=record,front_static_normal_load_N=float(scale),primary_ratio=float(750/scale)))
    # Nested tolerance decisions and the frozen headline are exact logical tests.
    headline=[next(r['reports'] for r in curves if r['population']=='primary_48' and r['stratum']=='all' and r['method']==k and r['tolerance_N']==750) for k in KEYS]
    assert headline==[31,84,222]
    for pop,setting in sorted(set((r['population'],r['setting']) for r in curves)):
        for k in KEYS:
            v=[r['reports'] for r in curves if r['population']==pop and r['setting']==setting and r['method']==k and r['stratum']=='all'];assert v==sorted(v)
    result=dict(protocol_sha256=hashlib.sha256((ROOT/'reproducibility/protocol_final_strengthening.json').read_bytes()).hexdigest(),curves=curves,run_level=runrows,
        primary_criterion_N=750,tolerance_grid_N=P['tolerance_grid_N'],normalization=scales,
        normalization_summary=dict(median_ratio=float(np.median([r['primary_ratio'] for r in scales])),min_ratio=min(r['primary_ratio'] for r in scales),max_ratio=max(r['primary_ratio'] for r in scales)),
        scope='Fixed intervals; 48 independent synthetic test trajectories or eight explicitly reused trajectories. Correlated within-run queries, no query bootstrap.')
    dump(ROOT/'results/final_tolerance_frontier.json',result)
    print(json.dumps([r for r in curves if r['stratum']=='all'],indent=2))
if __name__=='__main__':main()
