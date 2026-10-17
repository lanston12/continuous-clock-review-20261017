"""Frozen-protocol nonlinear holdout and retrospective CarSim cross-fit studies.
Test force labels are used in evaluation only, never in reconstruction inputs.
"""
from pathlib import Path
import numpy as np
import json, sys, hashlib, time, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from nonlinear_vehicle import specification,generate
from shared_fields import SharedFields,activity_edges
from shared_node_support import JointResponse,ClockOracle

_original_maximize=ClockOracle.maximize
def _cached_maximize(self,weights):
    """Positive homogeneity cache, used for every estimator in the comparison."""
    w=np.asarray(weights,float);scale=float(np.max(np.abs(w)));wn=w/scale if scale else w;key=tuple(wn)
    if not hasattr(self,'_homogeneous_cache'):self._homogeneous_cache={}
    if key not in self._homogeneous_cache:self._homogeneous_cache[key]=_original_maximize(self,wn)
    v=self._homogeneous_cache[key];return {**v,'value':v['value']*scale}
ClockOracle.maximize=_cached_maximize

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')
def protocol():return json.loads((ROOT/'reproducibility/protocol_v6.json').read_text())
def pl_integral(t,y,s,e):
    q=np.r_[s,t[(t>s)&(t<e)],e];return float(np.trapezoid(np.interp(q,t,y),q))
def interp_error(z,key,dt,start):
    t=z['t'];s=np.arange(start,t[-1]-.2,dt);v=np.interp(s,t,z[key]);q=np.unique(np.r_[s,t[(t>=s[0])&(t<=s[-1])]])
    q=q[(q>=.4)&(q<=t[-1]-.4)]
    return float(np.max(np.abs(np.interp(q,t,z[key])-np.interp(q,s,v))))
def activity_values(q,ta,ya,rho):
    ans=[]
    for v in q:
        i=max(0,np.searchsorted(ta,v-.08)-1);j=min(len(ta),np.searchsorted(ta,v+.08)+1)
        ans.append(max(abs(ya[i:j]))+.10+rho)
    return np.array(ans)
def calibrate(records,names):
    ra=1.25*max(interp_error(z,'A',.04,.205) for z in records);rr=1.25*max(interp_error(z,'r',.02,.2) for z in records);rows=[]
    for name,z in zip(names,records):
        t=z['t'];m,I,a,b=z['mechanics'];A=z['A'];r=z['r'];F=z['F'];T=a*F[:,0]-b*F[:,1];E=m*A-F.sum(1)
        sa=np.arange(.205,t[-1]-.2,.04);ta=sa-.025;ya=np.interp(sa,t,A);u=activity_values(t,ta,ya,ra);idx=(t>=.4)&(t<=t[-1]-.4)
        nl=I*np.diff(r)/np.diff(t)-T[:-1];nr=I*np.diff(r)/np.diff(t)-T[1:];mask=idx[:-1]&idx[1:]
        er=float(np.max(np.maximum(abs(E[idx])-25,0)/(m*u[idx])))
        lever=float(max(np.max(np.maximum(abs(nl[mask])-100,0)/(m*u[:-1][mask])),np.max(np.maximum(abs(nr[mask])-100,0)/(m*u[1:][mask]))))
        rows.append({'case':name,'force_ratio':er,'moment_lever_m':lever,'rho_a_raw':interp_error(z,'A',.04,.205),'rho_r_raw':interp_error(z,'r',.02,.2)})
    return {'rho_a_mps2':ra,'rho_r_rads':rr,'E_floor_N':25.,'N_floor_Nm':100.,'alpha_E':1.25*max(x['force_ratio'] for x in rows),'lever_N_m':1.25*max(x['moment_lever_m'] for x in rows),'inflation':1.25,'development_records':list(names),'calibration_rows':rows,'label_role':'development discrepancy-envelope calibration; test labels excluded'}
def make_record(z,seed,random_clocks=True):
    t=z['t'];g=np.random.default_rng(seed);sr=np.arange(.2,t[-1]-.2,.02);sa=np.arange(.205,t[-1]-.2,.04)
    dr,da=g.uniform(-.07,.07,2) if random_clocks else np.array([.035,-.025])
    er=g.uniform(-.002,.002,len(sr));ea=g.uniform(-.08,.08,len(sa));br=g.uniform(-.0005,.0005);ba=g.uniform(-.02,.02)
    yr=np.interp(sr,t,z['r'])+er+br;ya=np.interp(sa,t,z['A'])+ea+ba
    return (sr+dr,yr,sa+da,ya),{'dr':float(dr),'da':float(da),'sr':sr,'sa':sa,'er':er,'ea':ea,'br':float(br),'ba':float(ba)}
def crop(t,y,W):
    i=max(0,np.searchsorted(t,W.min()-.08)-1);j=min(len(t),np.searchsorted(t,W.max()+.08)+2)
    return t[i:j],y[i:j]
def model(z,rec,W,cfg):
    tr,yr,ta,ya=rec;tr,yr=crop(tr,yr,W);ta,ya=crop(ta,ya,W);m,I,a,b=z['mechanics'];mech=np.array([[.98*m,1.02*m],[.95*I,1.05*I],[a,a],[b,b]])
    M=SharedFields(tr,yr,ta,ya,W,mech,m,cfg)
    rel=JointResponse(tr,yr,ta,ya,W,[-.08,.08],[-.08,.08],mech,.002,.08,rho_r=cfg['rho_r_rads'],rho_a=cfg['rho_a_mps2'],E=M.E@M.J,N=M.N@M.J,bias_r=.0005,bias_a=.02)
    return M,rel

def targets(h):
    return [('front_mean_N',np.array([[1.,0.],[0.,0.]]),750.),('front_change_N',np.array([[-1.,0.],[1.,0.]]),750.),('axle_contrast_N',np.array([[1.,-1.],[0.,0.]]),750.),('total_impulse_Ns',np.array([[h,h],[0.,0.]]),250.)]

def query_record(name,z,rec,cfg,anchors,seed=0,metadata=None):
    rows=[]
    for s in anchors:
      for h in [.25,.5,1.]:
        W=np.array([[s,s+h],[s+h/2,s+1.5*h]])
        start=time.perf_counter();M,rel=model(z,rec,W,cfg);setup=time.perf_counter()-start
        marg=np.zeros((2,2,2));start=time.perf_counter()
        for j in range(2):
          for ax in range(2):
            unit=np.zeros((2,2));unit[j,ax]=1.;marg[j,ax]=M.interval(unit)[0]
        boxtime=time.perf_counter()-start
        for target,w,tol in targets(h):
          start=time.perf_counter();iv,lo,hi=M.interval(w);elapsed=time.perf_counter()-start
          start=time.perf_counter();riv=rel.interval(w)[0];rtime=time.perf_counter()-start
          box=np.array([np.sum(w*np.where(w>=0,marg[:,:,0],marg[:,:,1])),np.sum(w*np.where(w>=0,marg[:,:,1],marg[:,:,0]))])
          assert box[0]<=iv[0]+1e-5 and box[1]>=iv[1]-1e-5 and riv[0]<=iv[0]+1e-5 and riv[1]>=iv[1]-1e-5
          werr=max(abs(float(np.sum(w*lo['response']))-iv[0]),abs(float(np.sum(w*hi['response']))-iv[1]))
          row={'record':name,'seed':seed,'anchor_s':s,'window_s':h,'windows_s':W.tolist(),'weights':w.tolist(),'target':target,'units':'N s' if target.endswith('Ns') else 'N','tolerance':tol,'interval':iv.tolist(),'window_interval':riv.tolist(),'box_interval':box.tolist(),'width':float(iv[1]-iv[0]),'report':bool((iv[1]-iv[0])/2+1e-5<=tol),'witness_error':werr,'components':{k:(lo['components'][k]+hi['components'][k])/2 for k in lo['components']},'wall_s':{'setup':setup,'box_setup':boxtime,'shared_query':elapsed,'window_query':rtime}}
          if metadata:row.update(metadata)
          rows.append(row)
    # All intervals and conditional decisions are fixed before reading force labels.
    for row in rows:
        B=np.array([[pl_integral(z['t'],z['F'][:,ax],lo,hi)/(hi-lo) for ax in [0,1]] for lo,hi in row['windows_s']])
        truth=float(np.sum(np.array(row['weights'])*B));iv=row['interval']
        row.update(truth=truth,midpoint_error=abs(float(np.mean(iv))-truth),included=bool(iv[0]-1e-5<=truth<=iv[1]+1e-5))
    return rows

def applicability(z,rec,hidden,cfg):
    # Reference-only check after inference. PL validity is not a hardware guarantee.
    t=z['t'];A=z['A'];r=z['r'];F=z['F'];m,I,a,b=z['mechanics'];T=a*F[:,0]-b*F[:,1];lo=.4;hi=t[-1]-.4
    ed,u=activity_edges(rec[2],rec[3],lo,hi,cfg['rho_a_mps2']);q=np.unique(np.r_[ed,t[(t>lo)&(t<hi)]]);q=q[(q>=lo)&(q<=hi)]
    left=q[:-1];right=q[1:];mid=(left+right)/2;ue=u[np.minimum(np.searchsorted(ed,mid)-1,len(u)-1)];ix=np.minimum(np.searchsorted(t,mid)-1,len(t)-2)
    slope=I*np.diff(r)[ix]/np.diff(t)[ix];em=[];nm=[]
    for end in [left,right]:
        ff=np.interp(end,t,F[:,0]);fr=np.interp(end,t,F[:,1]);ev=m*np.interp(end,t,A)-ff-fr;nv=slope-a*ff+b*fr
        em.append(np.max(abs(ev)-(cfg['E_floor_N']+cfg['alpha_E']*m*ue)));nm.append(np.max(abs(nv)-(cfg['N_floor_Nm']+cfg['lever_N_m']*m*ue)))
    excess={'E':max(0.,float(max(em))),'N':max(0.,float(max(nm)))}
    for key,val,st,rho in [('rho_a',A,hidden['sa'],cfg['rho_a_mps2']),('rho_r',r,hidden['sr'],cfg['rho_r_rads'])]:
        qq=np.unique(np.r_[t[(t>=lo)&(t<=hi)],st[(st>=lo)&(st<=hi)]])
        diff=np.interp(qq,t,val)-np.interp(qq,st,np.interp(st,t,val));excess[key]=max(0.,float(max(abs(diff)))-rho)
    return {'full_record_reference_class_pass':all(v<=1e-6 for v in excess.values()),'bound_excess':excess}

def development():
    P=protocol();specs=[];records=[];names=[];dest=ROOT/'data/nonlinear_development';dest.mkdir(exist_ok=True)
    for i,seed in enumerate(P['development_seeds']):
        spec=specification(seed,i%4,0);z=generate(spec);name=f'dev_{seed}';np.savez_compressed(dest/f'{name}.npz',**z);specs.append(spec);records.append(z);names.append(name)
        print(name,'maxA',float(np.max(abs(z['A']))),'maxroll',float(np.max(abs(z['roll']))),flush=True)
    dump(ROOT/'data/development_specifications.json',specs);cfg=calibrate(records,names);dump(ROOT/'data/nonlinear_frozen_budget.json',cfg)
    lock={'protocol_sha256':hashlib.sha256((ROOT/'reproducibility/protocol_v6.json').read_bytes()).hexdigest(),'budget_sha256':hashlib.sha256((ROOT/'data/nonlinear_frozen_budget.json').read_bytes()).hexdigest(),'fit_records':len(records),'test_force_labels_used':False}
    dump(ROOT/'reproducibility/frozen_budget_manifest.json',lock);print(json.dumps(cfg,indent=2),flush=True)

def nonlinear_test():
    P=protocol();cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text());lock=json.loads((ROOT/'reproducibility/frozen_budget_manifest.json').read_text());assert lock['budget_sha256']==hashlib.sha256((ROOT/'data/nonlinear_frozen_budget.json').read_bytes()).hexdigest()
    rows=[];specs=[];apps=[];dest=ROOT/'data/nonlinear_test';dest.mkdir(exist_ok=True)
    for i in range(P['test_runs']):
        seed=P['test_seed_start']+i;spec=specification(seed,i%4,(i//4)%2);z=generate(spec);name=f'test_{seed}';np.savez_compressed(dest/f'{name}.npz',**z);specs.append(spec)
        rec,hidden=make_record(z,seed+P['noise_seed_offset'])
        r=query_record(name,z,rec,cfg,P['anchors_s'],seed,{'tire_law':spec['law'],'maneuver':spec['mode'],'split':'test'})
        ap=applicability(z,rec,hidden,cfg);apps.append({'record':name,**ap})
        for row in r:row['reference_class_pass']=ap['full_record_reference_class_pass']
        rows.extend(r)
        print(name,'law',spec['law'],'included',sum(x['included'] for x in r),'reports',sum(x['report'] for x in r),'class',ap['full_record_reference_class_pass'],flush=True)
        dump(ROOT/'results/nonlinear_test_rows.json',rows);dump(ROOT/'results/nonlinear_test_applicability.json',apps)
    dump(ROOT/'data/test_specifications.json',specs)
    refine=[]
    for ix in [0,17,35]:
        spec=specs[ix];z=dict(np.load(dest/f"test_{spec['seed']}.npz"));fine=generate(spec,dt=.001)
        diffs={key:float(np.max(np.abs(z[key]-np.interp(z['t'],fine['t'],fine[key])))) for key in ['A','r','roll']}
        diffs['max_force_N']=max(float(np.max(np.abs(z['F'][:,ax]-np.interp(z['t'],fine['t'],fine['F'][:,ax])))) for ax in [0,1]);refine.append({'record':f"test_{spec['seed']}",**diffs})
    dump(ROOT/'results/integration_refinement.json',refine)

def load_carsim(case):
    z=dict(np.load(ROOT/'data'/f'{case}.npz'));F=np.column_stack([sum(z['Fy_'+s+str(ax)]*np.cos(np.deg2rad(z['Steer_'+s+str(ax)]))+z['Fx_'+s+str(ax)]*np.sin(np.deg2rad(z['Steer_'+s+str(ax)])) for s in ['L','R']) for ax in [1,2]])
    return {'t':z['T_Stamp'],'A':z['Ay']*9.80665,'r':np.deg2rad(z['AVz']),'F':F,'mechanics':z['mechanics'],'steer':np.deg2rad((z['Steer_L1']+z['Steer_R1'])/2),'speed':z['Vx']/3.6}
def anchors(z):
    t=z['t'];u=np.rad2deg(z['steer']);eligible=[float(now) for i,now in enumerate(t) if .5<=now<=t[-1]-2 and z['speed'][i]>=5 and np.sqrt(np.mean(u[(t>=now-.25)&(t<=now)]**2))>=.5]
    return [] if not eligible else [eligible[0],eligible[0]+1]
def crossfit():
    cases=['dlc','dlc30','dlc40','sine_dwell','transient'];zs={c:load_carsim(c) for c in cases};rows=[];apps=[];budgets={}
    for case in cases:
      train=[c for c in cases if c!=case];cfg=calibrate([zs[c] for c in train],train);budgets[case]=cfg
      for seed in range(6100,6110):
        z=zs[case];rec,hidden=make_record(z,seed,random_clocks=False)
        rr=query_record(case,z,rec,cfg,anchors(z),seed,{'split':'retrospective_record_crossfit'})
        ap=applicability(z,rec,hidden,cfg)
        for row in rr:
            row['tolerance']=500. if row['units']=='N s' else 1500.;row['report']=row['width']/2+1e-5<=row['tolerance']
        rows+=rr;apps.append({'record':case,'seed':seed,**ap})
        print('CROSSFIT',case,seed,'inc',sum(x['included'] for x in rr),'report',sum(x['report'] for x in rr),'class',ap['full_record_reference_class_pass'],flush=True)
      dump(ROOT/'results/carsim_crossfit_rows.json',rows);dump(ROOT/'results/carsim_crossfit_applicability.json',apps);dump(ROOT/'results/carsim_crossfit_budgets.json',budgets)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','test','crossfit']);a=p.parse_args();{'development':development,'test':nonlinear_test,'crossfit':crossfit}[a.stage]()
if __name__=='__main__':main()
