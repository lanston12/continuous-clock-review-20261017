from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from sparse_clock import SparseClockOracle
from shared_fields import SharedFields,activity_edges
from experiments_v6 import make_record,dump,pl_integral
from shared_node_support import JointResponse
import shared_fields,shared_node_support
shared_fields.ClockOracle=SparseClockOracle;shared_node_support.ClockOracle=SparseClockOracle

def build(rec,W,mech,m0,cfg,clocks):
    tr,yr,ta,ya=rec
    def crop(t,y):
        i=max(0,np.searchsorted(t,W.min()-.08)-1);j=min(len(t),np.searchsorted(t,W.max()+.08)+2)
        return t[i:j],y[i:j]
    tr,yr=crop(tr,yr);ta,ya=crop(ta,ya)
    M=SharedFields(tr,yr,ta,ya,W,mech,m0,cfg)
    M.yaw=SparseClockOracle(tr,yr,W,clocks,.002,False,.0005)
    M.acc=SparseClockOracle(ta,ya,W,clocks,.08,True,.02)
    R=JointResponse(tr,yr,ta,ya,W,clocks,clocks,mech,.002,.08,
        rho_r=cfg['rho_r_rads'],rho_a=cfg['rho_a_mps2'],E=M.E@M.J,N=M.N@M.J,bias_r=.0005,bias_a=.02)
    return M,R

def membership(z,rec,hidden,cfg,mech,m0,clocks):
    # Uses force labels and injected times strictly after the interval computation.
    t,A,r,F=z['t'],z['A'],z['r'],z['F'];m,I,a,b=z['mechanics']
    lo,hi=.4,9.6;ed,u=activity_edges(rec[2],rec[3],lo,hi,cfg['rho_a_mps2'])
    q=np.unique(np.r_[ed,t[(t>lo)&(t<hi)]]);mid=(q[:-1]+q[1:])/2
    ue=u[np.minimum(np.searchsorted(ed,mid)-1,len(u)-1)];ix=np.minimum(np.searchsorted(t,mid)-1,len(t)-2)
    slope=I*np.diff(r)[ix]/np.diff(t)[ix];ex={'E':0.,'N':0.}
    for end in [q[:-1],q[1:]]:
        ff=np.interp(end,t,F[:,0]);fr=np.interp(end,t,F[:,1])
        ev=m*np.interp(end,t,A)-ff-fr;nv=slope-a*ff+b*fr
        ex['E']=max(ex['E'],float(np.max(abs(ev)-cfg['E_floor_N']-cfg['alpha_E']*m0*ue)))
        ex['N']=max(ex['N'],float(np.max(abs(nv)-cfg['N_floor_Nm']-cfg['lever_N_m']*m0*ue)))
    for key,val,st,rho in [('rho_a',A,hidden['sa'],cfg['rho_a_mps2']),('rho_r',r,hidden['sr'],cfg['rho_r_rads'])]:
        qq=np.unique(np.r_[t[(t>=lo)&(t<=hi)],st[(st>=lo)&(st<=hi)]])
        diff=np.interp(qq,t,val)-np.interp(qq,st,np.interp(st,t,val))
        ex[key]=max(0.,float(max(abs(diff)))-rho)
    mechanical=bool(np.all(z['mechanics']>=mech[:,0]) and np.all(z['mechanics']<=mech[:,1]))
    clock_ok=bool(all(clocks[0]<=hidden[k]<=clocks[1] for k in ['da','dr']))
    return dict(full_class_pass=mechanical and clock_ok and all(v<=1e-6 for v in ex.values()),
        mechanics_in_prior=mechanical,clocks_in_prior=clock_ok,residual_excess=ex)

def score(rec,mech,m0,cfg,clocks,anchors,durations):
    rows=[]
    for s in anchors:
        for h in durations:
            W=np.array([[s,s+h],[s+h/2,s+1.5*h]]);w=np.array([[-1.,0.],[1.,0.]])
            M,R=build(rec,W,mech,m0,cfg,clocks);iv=M.interval(w)[0];rv=R.interval(w)[0]
            rows.append(dict(anchor_s=s,duration_s=h,windows=W.tolist(),interval=iv.tolist(),window_interval=rv.tolist(),radius_N=float(np.diff(iv)[0]/2),report=bool(np.diff(iv)[0]/2+1e-5<=750)))
    return rows

def label_after(rows,z):
    for r in rows:
        W=np.array(r['windows']);v=[pl_integral(z['t'],z['F'][:,0],s,e)/(e-s) for s,e in W]
        truth=v[1]-v[0];r.update(truth_N=truth,included=r['interval'][0]-1e-5<=truth<=r['interval'][1]+1e-5,
            midpoint_error_N=abs(np.mean(r['interval'])-truth))

def measurements(z,seed,factor,half):
    g=np.random.default_rng(seed+100000);t=z['t'];sr=np.arange(.2,t[-1]-.2,.02*factor);sa=np.arange(.205,t[-1]-.2,.04*factor)
    dr,da=g.uniform(-.875*half,.875*half,2);er=g.uniform(-.002,.002,len(sr));ea=g.uniform(-.08,.08,len(sa));br=g.uniform(-.0005,.0005);ba=g.uniform(-.02,.02)
    rec=(sr+dr,np.interp(sr,t,z['r'])+er+br,sa+da,np.interp(sa,t,z['A'])+ea+ba)
    return rec,dict(sr=sr,sa=sa,dr=float(dr),da=float(da))

def main():
    protocol_path=ROOT/'reproducibility/protocol_v7_sensitivity.json';P=json.loads(protocol_path.read_text());cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text())
    allrows=[];apps=[];frontier=[];frontapps=[]
    for seed in P['records']:
        z=dict(np.load(ROOT/'data/nonlinear_test'/f'test_{seed}.npz'));true=z['mechanics'];m,I,a,b=true
        rec,hidden=make_record(z,seed+100000);base=np.array([[.98*m,1.02*m],[.95*I,1.05*I],[a,a],[b,b]])
        for setting in P['mechanics_cases']:
            mech=base.copy();m0=m;clocks=[-.08,.08];rr=tuple(x.copy() for x in rec);hh={**hidden}
            if setting=='mass_center_plus_5pct':mech[0]*=1.05;m0*=1.05
            elif setting=='inertia_center_plus_10pct':mech[1]*=1.10
            elif setting=='geometry_center_plus_3pct':mech[2:]*=1.03
            elif setting=='broader_mechanics':mech=np.array([[v*(1-q),v*(1+q)] for v,q in zip(true,P['broader_mechanics_halfwidths'])])
            elif setting=='clock_center_plus_40ms_half_40ms':clocks=[0.,.08]
            elif setting=='observation_gap':
                for index,stkey in [(0,'sr'),(2,'sa')]:
                    keep=(hh[stkey]<P['gap_physical_s'][0])|(hh[stkey]>P['gap_physical_s'][1])
                    seq=list(rr);seq[index]=rr[index][keep];seq[index+1]=rr[index+1][keep];rr=tuple(seq);hh[stkey]=hh[stkey][keep]
            rows=score(rr,mech,m0,cfg,clocks,P['anchors_s'],P['durations_s'])
            app=membership(z,rr,hh,cfg,mech,m0,clocks);label_after(rows,z)
            for r in rows:r.update(record=f'test_{seed}',setting=setting,**app)
            allrows+=rows;apps.append(dict(record=f'test_{seed}',setting=setting,**app))
        for factor in P['sampling_factors']:
            for half in P['clock_half_ranges_s']:
                rr,hh=measurements(z,seed,factor,half);clocks=[-half,half]
                rows=score(rr,base,m,cfg,clocks,P['anchors_s'],P['durations_s'])
                app=membership(z,rr,hh,cfg,base,m,clocks);label_after(rows,z)
                for r in rows:r.update(record=f'test_{seed}',sampling_factor=factor,clock_half_s=half,**app)
                frontier+=rows;frontapps.append(dict(record=f'test_{seed}',sampling_factor=factor,clock_half_s=half,**app))
        print('V7 SENSITIVITY',seed,flush=True)
        dump(ROOT/'results/v7_sensitivity.json',dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),rows=allrows,applicability=apps))
        dump(ROOT/'results/v7_resolution_frontier.json',dict(rows=frontier,applicability=frontapps))

if __name__=='__main__':main()
