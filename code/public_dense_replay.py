"""Independent raw-dense replay of synthetic queries only; no restricted inputs."""
from pathlib import Path
import json,numpy as np
from experiments_v6 import model,make_record,dump
from dense_raw_clock import RawDenseClockOracle
import shared_fields,shared_node_support
ROOT=Path(__file__).resolve().parents[1]
def main():
    shared_fields.ClockOracle=RawDenseClockOracle;shared_node_support.ClockOracle=RawDenseClockOracle
    cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text());rows=json.loads((ROOT/'results/nonlinear_test_rows.json').read_text())
    out=[];gap=0.;changes=0
    for record in sorted(set(r['record'] for r in rows)):
        selected=[r for r in rows if r['record']==record];z=dict(np.load(ROOT/'data/nonlinear_test'/(record+'.npz')))
        rec,_=make_record(z,selected[0]['seed']+100000)
        for anchor,h in sorted(set((r['anchor_s'],r['window_s']) for r in selected)):
            W=np.array([[anchor,anchor+h],[anchor+h/2,anchor+1.5*h]]);M,_=model(z,rec,W,cfg)
            for r in selected:
                if r['anchor_s']!=anchor or r['window_s']!=h:continue
                iv=M.interval(r['weights'])[0];delta=float(max(abs(iv-r['interval'])))
                changed=bool(np.diff(iv)[0]/2+1e-5<=r['tolerance'])!=r['report'];assert delta<1e-6 and not changed
                gap=max(gap,delta);changes+=changed
                out.append(dict(record=record,anchor_s=anchor,duration_s=h,target=r['target'],interval=iv.tolist(),gap_N_or_Ns=delta))
        print('SYNTHETIC RAW REPLAY',record,flush=True)
    assert len(out)==1728
    dump(ROOT/'results/public_dense_replay.json',dict(passed=True,queries=len(out),max_interval_gap=gap,decision_changes=changes,rows=out))
if __name__=='__main__':main()
