from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from dense_raw_clock import RawDenseClockOracle
from experiments_v6 import make_record,model,targets,dump
import shared_fields,shared_node_support
shared_fields.ClockOracle=RawDenseClockOracle;shared_node_support.ClockOracle=RawDenseClockOracle
rows=[]
cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text())
for seed in range(83000,83008):
    z=dict(np.load(ROOT/'data/nonlinear_test'/f'test_{seed}.npz'));rec,_=make_record(z,seed+100000)
    W=np.array([[2.,2.5],[2.25,2.75]]);M,_=model(z,rec,W,cfg)
    for target,w,tol in targets(.5):
        exact=M.interval(w)[0];start=time.perf_counter()
        try:
            generic,lo,hi=M.interval(w,solver='bernstein')
            gap=float(max(exact[0]-generic[0],generic[1]-exact[1]));contain=bool(generic[0]<=exact[0]+1e-6 and generic[1]>=exact[1]-1e-6)
            assert contain and gap<.01
            rows.append(dict(record=f'test_{seed}',target=target,units='N s' if target.endswith('Ns') else 'N',exact=exact.tolist(),generic_outer=generic.tolist(),max_outer_gap=gap,status='resolved',wall_s=time.perf_counter()-start))
        except FloatingPointError as e:
            rows.append(dict(record=f'test_{seed}',target=target,status='numerically_unresolved',reason=str(e),wall_s=time.perf_counter()-start))
    print('ENGINEERING GENERIC',seed,flush=True)
dump(ROOT/'results/v7_generic_force.json',dict(rows=rows,queries=len(rows),resolved=sum(r['status']=='resolved' for r in rows),max_outer_gap=max([r.get('max_outer_gap',0.) for r in rows]),max_gap_by_units={u:max(r['max_outer_gap'] for r in rows if r.get('units')==u) for u in ['N','N s']},scope='Same measurements, windows, mechanics, fields and engineering units; numerical Bernstein enclosure, not a CPZ/SPZ software replication'))
