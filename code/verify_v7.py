from pathlib import Path
import sys,json,time,platform,tracemalloc,gc,decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from dense_raw_clock import RawDenseClockOracle,raw_operator
from sparse_clock import SparseClockOracle,_maximize,_roots_inside
from shared_node_support import ClockOracle,operator
from experiments_v6 import dump,make_record,model,load_carsim
import shared_fields,shared_node_support

def verify():
    rng=np.random.default_rng(761016);rows=[];image_gap=0.;point_gap=0.
    for k in range(24):
        n=[35,81,160,301][k%4];t=np.r_[0,np.cumsum(rng.uniform(.012,.065,n-1))]
        y=rng.normal(size=n);J=[1,2,4,6][k%4];width=min(.3,t[-1]/5)
        starts=rng.uniform(.12,t[-1]-width-.12,J);W=np.c_[starts,starts+width]
        if k%4==1:W[1]=W[0] # identical windows
        if k%4==2:W[1]=[W[0,0]+width/4,W[0,1]-width/4] # nesting
        clock=[-.04,.055] if k%6 else [.011,.011]
        eps=rng.uniform(0,.03,n);eps[::7]=0
        for average in [True,False]:
            S=SparseClockOracle(t,y,W,clock,eps,average,.02)
            D=RawDenseClockOracle(t,y,W,clock,eps,average,.02)
            ref=ClockOracle(t,y,W,clock,eps,average,.02)
            assert np.array_equal(S.edges,D.edges)
            for j in range(3):
                w=rng.normal(size=J);sv=S.maximize(w);dv=D.maximize(w);rv=ref.maximize(w)
                gap=max(abs(sv['value']-dv['value']),abs(dv['value']-rv['value']))
                assert gap<=1e-7*(1+abs(sv['value']))
                rows.append(dict(case=k,average=average,windows=J,nodes=n,gap=gap,witness_gap=dv['witness_gap']))
            # Arbitrary interior-node lifting, not just a supporting direction.
            area=np.r_[np.diff(t)/2,0]+np.r_[0,np.diff(t)/2]
            for kk in np.unique(np.linspace(0,len(S.edges)-2,min(4,len(S.edges)-1)).astype(int)):
                lo,hi=S.edges[kk:kk+2];d=lo+.371*(hi-lo);x=.371 if hi>lo else 0.
                Q,C,AE,AN,SS,BB,SN=S.prepared
                mid=(lo+hi)/2;active=np.unique(np.array([[max(0,min(n-2,np.searchsorted(t,z+mid,side='right')-1)),max(0,min(n-2,np.searchsorted(t,z+mid,side='right')-1))+1] for z in W.ravel()]).ravel())
                err=np.zeros(n);err[active]=rng.uniform(-1,1,len(active))*eps[active]
                z=rng.uniform(-1,1,SN[kk])*BB[kk,:SN[kk]]
                ends=np.sort(np.unique(W.ravel()))+mid
                for i in range(n):
                    if i in active:continue
                    u=np.searchsorted(ends,t[i],side='right')-1
                    if 0<=u<SN[kk] and BB[kk,u]>0:err[i]=eps[i]*z[u]/BB[kk,u]
                powers=np.array([1.,x,x*x]);compact=Q[kk]@powers+C[kk,:,:len(active),:]@powers@err[active]
                if average:compact+=SS[kk,:SN[kk]].T@z
                exact=raw_operator(t,W,d,average)@(y+err)
                image_gap=max(image_gap,float(max(abs(exact-compact))))
                assert np.max(abs(exact-compact))<1e-8
                # Independent direct PL trapezoid/endpoint reference.
                for jj,(s,e) in enumerate(W):
                    u,v=s+d,e+d
                    if average:
                        knots=np.r_[u,t[(t>u)&(t<v)],v]
                        direct=np.trapezoid(np.interp(knots,t,y+err),knots)/(e-s)
                    else:direct=(np.interp(v,t,y+err)-np.interp(u,t,y+err))/(e-s)
                    point_gap=max(point_gap,abs(direct-exact[jj]));assert abs(direct-exact[jj])<1e-8
    # High precision root checks use exact Decimal representations of float64 coefficients.
    D=decimal.Decimal;decimal.getcontext().prec=80;root_rows=[]
    for coeff in [[.25,-1.,1.],[0.,0.,0.],[-.5,1.,0.],[.25-1e-14,-1.,1.],[.25+1e-14,-1.,1.],[1e-30,-1e-15,1.]]:
        c,b,a=map(D.from_float,coeff);disc=b*b-4*a*c;expected=[]
        if a:
            if disc>=0:expected=[float((-b-disc.sqrt())/(2*a)),float((-b+disc.sqrt())/(2*a))]
        elif b:expected=[float(-c/b)]
        expected=sorted(x for x in expected if 0<x<1)
        out=np.empty(4);nroot=_roots_inside(*coeff,out,0);actual=sorted(out[:nroot].tolist())
        resolved=len(expected)==len(actual) and (not actual or max(abs(np.array(expected)-actual))<1e-8)
        root_rows.append(dict(coeff=coeff,high_precision_roots=expected,float64_roots=actual,status='resolved' if resolved else 'numerically_unresolved'))
    assert all(x['status']=='resolved' for x in root_rows)
    rejected=[];base=dict(t=np.array([0.,1.,2.]),y=np.zeros(3),windows=np.array([[.5,1.5]]),clock=[0.,0.],eps=.1)
    for label,change in [('duplicate_nodes',dict(t=np.array([0.,0.,2.]))),('nonfinite',dict(y=np.array([0.,np.nan,0.]))),('zero_window',dict(windows=np.array([[1.,1.]]))),('extrapolation',dict(clock=[-1.,1.])),('negative_errors',dict(eps=-.1)),('reversed_clock',dict(clock=[.1,-.1]))]:
        try:RawDenseClockOracle(**{**base,**change})
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid input accepted: '+label)
    dump(ROOT/'results/v7_dense_verification.json',dict(passed=True,checks=len(rows),max_support_gap=max(x['gap'] for x in rows),arbitrary_lifting_max_gap=image_gap,direct_PL_max_gap=point_gap,rows=rows,high_precision_root_diagnostics=root_rows,rejected_invalid_inputs=rejected))
    print('V7 independent dense verification',len(rows),image_gap,point_gap,flush=True)

def replay():
    shared_fields.ClockOracle=RawDenseClockOracle;shared_node_support.ClockOracle=RawDenseClockOracle
    cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text());maxgap=0.;changes=0;allrows=[]
    budgets=json.loads((ROOT/'results/carsim_crossfit_budgets.json').read_text())
    for study,file in [('nonlinear','nonlinear_test_rows.json'),('carsim','carsim_crossfit_rows.json')]:
        rows=json.loads((ROOT/'results'/file).read_text())
        for name in sorted(set(r['record'] for r in rows)):
            selected=[r for r in rows if r['record']==name]
            z=dict(np.load(ROOT/'data/nonlinear_test'/f'{name}.npz')) if study=='nonlinear' else load_carsim(name)
            for seed in sorted(set(r['seed'] for r in selected)):
                rec,_=make_record(z,seed+100000,True) if study=='nonlinear' else make_record(z,seed,False)
                for anchor,h in sorted(set((r['anchor_s'],r['window_s']) for r in selected if r['seed']==seed)):
                    W=np.array([[anchor,anchor+h],[anchor+h/2,anchor+1.5*h]])
                    M,_=model(z,rec,W,cfg if study=='nonlinear' else budgets[name])
                    for r in [r for r in selected if r['seed']==seed and r['anchor_s']==anchor and r['window_s']==h]:
                        iv,lo,hi=M.interval(r['weights']);gap=float(max(abs(iv-r['interval'])))
                        changed=bool(np.diff(iv)[0]/2+1e-5<=r['tolerance'])!=r['report']
                        maxgap=max(maxgap,gap);changes+=int(changed)
                        assert gap<1e-6 and not changed
                        allrows.append(dict(study=study,record=name,seed=seed,anchor=anchor,h=h,target=r['target'],interval=iv.tolist(),gap=gap))
            print('RAW DENSE REPLAY',study,name,flush=True)
    dump(ROOT/'results/v7_raw_dense_replay.json',dict(passed=True,queries=len(allrows),max_interval_gap=maxgap,decision_changes=changes,rows=allrows))

def witness(oracle,w,d):
    op=raw_operator(oracle.t,oracle.windows,d,oracle.average);c=w@op
    return op@(oracle.y+np.sign(c)*oracle.eps+(np.sign(c.sum())*oracle.bias if oracle.average else 0.))

def benchmark():
    rng=np.random.default_rng(39116);rows=[]
    t=np.linspace(0,2,101);W=np.array([[.4,.9],[.7,1.2]])
    for typ in [SparseClockOracle,RawDenseClockOracle]:
        o=typ(t,np.sin(t),W,[-.08,.08],.01);_maximize(*o.prepared,o.edges,np.array([1.,-1.]),o.bias,o.average);witness(o,np.array([1.,-1.]),0.)
    for J in [2,4,8]:
      for n in [128,256,512,1024,2048]:
        t=np.linspace(0,3,n);W=np.c_[np.linspace(.45,1.45,J),np.linspace(.45,1.45,J)+.8]
        ix=(t>=W.min()-.12)&(t<=W.max()+.12);t=t[ix];y=np.sin(3.3*t)+.2*np.cos(8.1*t);eps=.012+.006*(1+np.sin(4*t))
        directions=rng.normal(size=(32,J));directions/=np.max(abs(directions),axis=1)[:,None]
        measured={};maxgap=0.
        for label,typ in [('compressed',SparseClockOracle),('raw_dense',RawDenseClockOracle)]:
            setup=[];timings={str(b):[] for b in [1,16,32]};wit=[];answers=None
            for repeat in range(5):
                gc.collect();start=time.perf_counter();o=typ(t,y,W,[-.08,.08],eps,True,.015);setup.append(time.perf_counter()-start)
                for batch in [1,16,32]:
                    start=time.perf_counter();result=[_maximize(*o.prepared,o.edges,w,o.bias,o.average) for w in directions[:batch]];timings[str(batch)].append(time.perf_counter()-start)
                answers=np.array([r[0] for r in result]);start=time.perf_counter()
                decoded=[witness(o,w,r[1]) for w,r in zip(directions,result)];wit.append(time.perf_counter()-start)
                assert max(abs(w@v-r[0]) for w,v,r in zip(directions,decoded,result))<1e-7
            del o;gc.collect();tracemalloc.start();o=typ(t,y,W,[-.08,.08],eps,True,.015);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            def stat(a):return dict(median=float(np.median(a)),min=float(min(a)),max=float(max(a)),samples=a)
            measured[label]=dict(setup=stat(setup),queries={b:stat(a) for b,a in timings.items()},witness_32=stat(wit),payload_bytes=o.storage_bytes(),traced_peak_bytes=peak)
            if label=='compressed':reference=answers
            else:maxgap=float(max(abs(reference-answers)));assert maxgap<1e-7
            del o
        row=dict(nodes_before_crop=n,nodes=len(t),windows=J,max_support_disagreement=maxgap,measurements=measured)
        rows.append(row);print('V7 BENCHMARK',J,n,'gap',maxgap,flush=True)
    dump(ROOT/'results/v7_independent_benchmark.json',dict(rows=rows,repeats=5,warmup_excluded=True,same_kernel=True,witness_decoding='same independent raw operator; separately timed',memory='numeric payload and traced allocation peak; not total process RSS',environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),cpu=platform.processor())))

if __name__=='__main__':
    {'verify':verify,'replay':replay,'benchmark':benchmark}[sys.argv[1]]()
