"""Frozen eight-record same-information comparison; force labels enter scoring only."""
from pathlib import Path
import hashlib,json
import numpy as np
from sensitivity_v7 import build,membership
from experiments_v6 import make_record,dump,pl_integral
ROOT=Path(__file__).resolve().parents[1]
METHODS=['box','window','shared']

def infer(rec,mechanics,nominal_mass,cfg,P):
    rows=[]
    for s in P['anchors_s']:
        for h in P['durations_s']:
            W=np.array([[s,s+h],[s+h/2,s+1.5*h]])
            w=np.array([[-1.,0.],[1.,0.]])
            M,R=build(rec,W,mechanics,nominal_mass,cfg,[-.08,.08])
            marginal=np.empty((2,2,2))
            for j in range(2):
                for k in range(2):
                    u=np.zeros((2,2));u[j,k]=1
                    marginal[j,k]=M.interval(u)[0]
            box=np.array([np.sum(w*np.where(w>=0,marginal[:,:,0],marginal[:,:,1])),
                          np.sum(w*np.where(w>=0,marginal[:,:,1],marginal[:,:,0]))])
            ivs=dict(box=box,window=R.interval(w)[0],shared=M.interval(w)[0])
            rows.append(dict(anchor_s=s,duration_s=h,windows_s=W.tolist(),
                component_marginals_N=marginal.tolist(),intervals_N={k:v.tolist() for k,v in ivs.items()},
                envelope_edges_s=M.edges.tolist() if hasattr(M,'edges') else [],
                integrated_force_budget_N=(M.E@M.J).tolist(),integrated_moment_budget_Nm=(M.N@M.J).tolist(),
                radius_N={k:float(np.diff(v)[0]/2) for k,v in ivs.items()},
                reports={k:bool(np.diff(v)[0]/2+P['decision_guard_N']<=750) for k,v in ivs.items()}))
    return rows

def main():
    protocol=ROOT/'reproducibility/protocol_final_strengthening.json';P=json.loads(protocol.read_text())
    cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text())
    old=json.loads((ROOT/'results/v7_sensitivity.json').read_text())['rows']
    rows=[];runs=[];violations=[];max_saved_gap=0.;max_leak=0.;decision_changes=0;inputs=[]
    for seed in P['records']:
        path=ROOT/'data/nonlinear_test'/f'test_{seed}.npz';z=dict(np.load(path));true=z['mechanics']
        rec,hidden=make_record(z,seed+100000)
        sensor_hash=hashlib.sha256(b''.join(x.tobytes() for x in rec)).hexdigest()
        canonical=hashlib.sha256(b''.join(k.encode()+np.ascontiguousarray(z[k]).tobytes() for k in sorted(z))).hexdigest()
        inputs.append(dict(record=f'test_{seed}',native_array_sha256=canonical,sensor_sha256=sensor_hash))
        for setting,spec in P['priors'].items():
            center=true*np.array(spec['center_multiplier']);width=np.array(spec['relative_halfwidth'])
            mech=np.column_stack((center*(1-width),center*(1+width)))
            # The record's fixed nominal mass, hence every discrepancy envelope, is unchanged.
            raw=infer(rec,mech,float(true[0]),cfg,P)
            # Full input-wrapper mutation, rebuilding record, declared prior and all decisions.
            bad={k:v.copy() for k,v in z.items()}
            for key in ['F','E','N','roll','roll_rate','vy']:bad[key]=np.full_like(bad[key],1e8)
            badrec,_=make_record(bad,seed+100000)
            badcenter=bad['mechanics']*np.array(spec['center_multiplier'])
            badmech=np.column_stack((badcenter*(1-width),badcenter*(1+width)))
            mutated=infer(badrec,badmech,float(bad['mechanics'][0]),cfg,P)
            for orig,mut in zip(raw,mutated):
                for method in METHODS:
                    max_leak=max(max_leak,float(np.max(abs(np.array(orig['intervals_N'][method])-mut['intervals_N'][method]))))
                    decision_changes+=orig['reports'][method]!=mut['reports'][method]
            # Evaluation only, after every interval and report decision is fixed.
            app=membership(z,rec,hidden,cfg,mech,float(true[0]),[-.08,.08])
            record=f'test_{seed}';mode=(seed-83000)%4;law=((seed-83000)//4)%2
            runrows=[]
            for item in raw:
                W=item['windows_s'];truth=pl_integral(z['t'],z['F'][:,0],*W[1])/(W[1][1]-W[1][0])-pl_integral(z['t'],z['F'][:,0],*W[0])/(W[0][1]-W[0][0])
                ivs=item['intervals_N']
                for outer,inner in [('box','window'),('window','shared'),('box','shared')]:
                    excess=max(ivs[outer][0]-ivs[inner][0],ivs[inner][1]-ivs[outer][1],0.)
                    if excess>P['nesting_tolerance_N']:violations.append(dict(record=record,setting=setting,anchor=item['anchor_s'],duration=item['duration_s'],outer=outer,inner=inner,excess_N=excess))
                legacy='original_known_vehicle' if setting=='known_vehicle' else 'broader_mechanics' if setting=='wide_centered' else None
                if legacy:
                    ref=next(r for r in old if r['record']==record and r['setting']==legacy and r['anchor_s']==item['anchor_s'] and r['duration_s']==item['duration_s'])
                    for method,key in [('shared','interval'),('window','window_interval')]:max_saved_gap=max(max_saved_gap,float(np.max(abs(np.array(ivs[method])-ref[key]))))
                for method in METHODS:
                    iv=ivs[method];rad=item['radius_N'][method]
                    row=dict(record=record,setting=setting,method=method,mode=mode,tire_law=law,
                        anchor_s=item['anchor_s'],duration_s=item['duration_s'],windows_s=W,
                        mechanics_center=center.tolist(),mechanics_intervals=mech.tolist(),nominal_envelope_mass_kg=float(true[0]),
                        sensor_sha256=sensor_hash,clock_ranges_s=[[-.08,.08],[-.08,.08]],
                        interval_N=iv,width_N=2*rad,radius_N=rad,report=item['reports'][method],
                        decision='report' if item['reports'][method] else 'unknown',
                        truth_N=float(truth),midpoint_error_N=float(abs(np.mean(iv)-truth)),
                        truth_contained=bool(iv[0]-1e-5<=truth<=iv[1]+1e-5),**app)
                    rows.append(row);runrows.append(row)
            joint=np.array([r['width_N'] for r in runrows if r['method']=='shared']);ind=np.array([r['width_N'] for r in runrows if r['method']=='window'])
            runs.append(dict(record=record,setting=setting,mode=mode,tire_law=law,queries_per_method=9,**app,
                reports={k:sum(r['report'] for r in runrows if r['method']==k) for k in METHODS},
                mean_radius_N={k:float(np.mean([r['radius_N'] for r in runrows if r['method']==k])) for k in METHODS},
                mean_width_reduction=1-float(np.mean(joint)/np.mean(ind)),
                report_gain_pp=100*sum(r['report'] for r in runrows if r['method']=='shared')/9-100*sum(r['report'] for r in runrows if r['method']=='window')/9))
        print('FINAL PAIRED',seed,flush=True)
    summaries=[]
    for setting in P['priors']:
        rr=[r for r in rows if r['setting']==setting];run=[r for r in runs if r['setting']==setting]
        summaries.append(dict(setting=setting,independent_runs=len(run),queries_per_method=72,class_valid_runs=sum(r['full_class_pass'] for r in run),
            reports={k:sum(r['report'] for r in rr if r['method']==k) for k in METHODS},
            median_radius_N={k:float(np.median([r['radius_N'] for r in rr if r['method']==k])) for k in METHODS},
            truth_failures={k:sum(not r['truth_contained'] for r in rr if r['method']==k) for k in METHODS},
            run_mean_width_reduction=float(np.mean([r['mean_width_reduction'] for r in run])),
            run_report_gain_pp=[r['report_gain_pp'] for r in run]))
    result=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),frozen_budget_sha256=hashlib.sha256((ROOT/'data/nonlinear_frozen_budget.json').read_bytes()).hexdigest(),
        inputs=inputs,rows=rows,run_summaries=runs,summary=summaries,nesting_violations=violations,numerical_failures=[],
        max_legacy_interval_difference_N=max_saved_gap,label_mutation_max_interval_change_N=max_leak,label_mutation_decision_changes=decision_changes,
        label_mutation_queries=8*3*9*3,scope='Eight reused test trajectories, three same-information priors; force labels used only after inference.')
    dump(ROOT/'results/final_paired_priors.json',result)
    assert not violations and max_saved_gap<1e-7 and max_leak==0 and decision_changes==0
    assert next(s for s in summaries if s['setting']=='wide_centered')['reports']['shared']==28
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
