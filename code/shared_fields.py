"""Continuous clocks and shared activity-weighted balance discrepancies.

Real-arithmetic finite support; float64 implementation is not a rounding certificate.
The empirical envelope is fixed from development, not a physical universal bound.
"""
import itertools,heapq
import numpy as np
from shared_node_support import ClockOracle,operator

def validate(*arrays):
    if any(not np.all(np.isfinite(np.asarray(x,float))) for x in arrays):raise ValueError('Nonfinite input')

def activity_edges(ta,ya,lo,hi,rho):
    edges=np.unique(np.r_[lo,hi,(ta-.08)[(ta-.08>lo)&(ta-.08<hi)],(ta+.08)[(ta+.08>lo)&(ta+.08<hi)]])
    u=[]
    for x in (edges[:-1]+edges[1:])/2:
        i=max(0,np.searchsorted(ta,x-.08)-1);j=min(len(ta),np.searchsorted(ta,x+.08)+1)
        u.append(max(abs(ya[i:j]))+.10+rho)
    return edges,np.asarray(u)

def bernstein_max(oracle,weights,tol=1e-7):
    """Generic dependency-preserving polynomial B&B, not a CPZ replication.

    Upper bound follows convexity of q+sum eps*abs(c) on a common Bernstein hull.
    """
    best=-np.inf;heap=[];serial=0;calls=0
    def bounds(c,l,h):
        nonlocal best,calls
        xs=np.array([l,(l+h)/2,h]);V=c@np.array([np.ones(3),xs,xs*xs]);v=oracle.y@V+oracle.eps@abs(V)+oracle.bias*abs(V.sum(0));best=max(best,float(max(v)));calls+=3
        B=np.column_stack([V[:,0],2*V[:,1]-(V[:,0]+V[:,2])/2,V[:,2]])
        ub=float(max(oracle.y@B+oracle.eps@abs(B)+oracle.bias*abs(B.sum(0))))
        return ub
    for _,_,P in oracle.cells:
        c=np.einsum('j,jnp->np',weights,P);ub=bounds(c,0,1);heapq.heappush(heap,(-ub,serial,0.,1.,c));serial+=1
    while heap and -heap[0][0]-best>tol and calls<100000:
        neg,_,l,h,c=heapq.heappop(heap);mid=(l+h)/2
        for ll,hh in [(l,mid),(mid,h)]:
            ub=bounds(c,ll,hh)
            if ub>best:heapq.heappush(heap,(-ub,serial,ll,hh,c));serial+=1
    upper=max(best,-heap[0][0]) if heap else best
    return {'value':upper,'lower':best,'gap':upper-best,'evaluations':calls,'status':'numerical_ok' if upper-best<=tol else 'numerical_unknown'}

class SharedFields:
    def __init__(self,tr,yr,ta,ya,windows,mechanics,nominal_mass,budget):
        validate(tr,yr,ta,ya,windows,mechanics,nominal_mass,list(budget.values()) if all(np.isscalar(v) for v in budget.values()) else [budget[k] for k in ['rho_a_mps2','rho_r_rads','E_floor_N','N_floor_Nm','alpha_E','lever_N_m']])
        self.windows=np.asarray(windows,float);self.mechanics=np.asarray(mechanics,float);self.mass=float(nominal_mass);self.budget=budget
        if self.mass<=0:raise ValueError('Positive nominal mass')
        if self.windows.ndim!=2 or self.windows.shape[1]!=2:raise ValueError('Window shape')
        if self.mechanics.shape!=(4,2) or np.any(self.mechanics<=0) or np.any(np.diff(self.mechanics,axis=1)<0):raise ValueError('Mechanics box')
        if any(budget[k]<0 for k in ['rho_a_mps2','rho_r_rads','E_floor_N','N_floor_Nm','alpha_E','lever_N_m']):raise ValueError('Negative budget')
        self.acc=ClockOracle(ta,ya,windows,[-.08,.08],.08,True,.02)
        self.yaw=ClockOracle(tr,yr,windows,[-.08,.08],.002,False,.0005)
        lo=self.windows[:,0].min();hi=self.windows[:,1].max();ed,u=activity_edges(np.asarray(ta),np.asarray(ya),lo,hi,budget['rho_a_mps2'])
        self.edges=np.unique(np.r_[ed,self.windows.ravel()]);mid=(self.edges[:-1]+self.edges[1:])/2;self.u=u[np.minimum(np.searchsorted(ed,mid)-1,len(u)-1)]
        self.J=((mid[:,None]>=self.windows[:,0])&(mid[:,None]<self.windows[:,1]))*np.diff(self.edges)[:,None]/np.diff(self.windows,axis=1)[:,0]
        self.E=budget['E_floor_N']+budget['alpha_E']*self.mass*self.u
        self.N=budget['N_floor_Nm']+budget['lever_N_m']*self.mass*self.u
        self.endpoint_times=np.unique(self.windows.ravel());self.D=np.zeros((len(self.endpoint_times),len(self.windows)))
        for j,(s,e) in enumerate(self.windows):
            self.D[np.searchsorted(self.endpoint_times,s),j]-=1/(e-s);self.D[np.searchsorted(self.endpoint_times,e),j]+=1/(e-s)

    def support(self,w,solver='exact'):
        w=np.asarray(w,float);validate(w)
        if w.shape!=(len(self.windows),2):raise ValueError('Weight shape')
        best=None
        for a,b in set(itertools.product(*self.mechanics[2:])):
            L=a+b;cA=(b*w[:,0]+a*w[:,1])/L;cR=(w[:,0]-w[:,1])/L;KA=self.J@cA;KN=self.J@cR;KD=self.D@cR
            av=self.acc.maximize(cA);rv=self.yaw.maximize(cR)
            if solver=='bernstein':
                ag=bernstein_max(self.acc,cA);rg=bernstein_max(self.yaw,cR)
                if ag['status']!='numerical_ok' or rg['status']!='numerical_ok':raise FloatingPointError('Generic subdivision limit: numerical unknown')
            for m,I in set(itertools.product(*self.mechanics[:2])):
                ea=np.sign(KA)*self.budget['rho_a_mps2'];er=np.sign(KD)*self.budget['rho_r_rads'];ef=-np.sign(KA)*self.E;en=-np.sign(KN)*self.N
                A=av['values']+ea@self.J;D=rv['values']+er@self.D;F=m*A-ef@self.J;Q=I*D-en@self.J
                response=np.c_[(b*F+Q)/L,(a*F-Q)/L]
                value=float(np.sum(w*response))
                if solver=='bernstein':value+=m*(ag['value']-av['value'])+I*(rg['value']-rv['value'])
                if best is None or value>best['value']:
                    # Additive decomposition at THIS witness; no sum of independent ablations.
                    acc_cv=cA@operator(self.acc.t,self.windows,av['clock'],True);yaw_cv=cR@operator(self.yaw.t,self.windows,rv['clock'],False)
                    zero=m*float(cA@operator(self.acc.t,self.windows,0,True)@self.acc.y)+I*float(cR@operator(self.yaw.t,self.windows,0,False)@self.yaw.y)
                    deterministic=m*float(acc_cv@self.acc.y)+I*float(yaw_cv@self.yaw.y)
                    mn,In,an,bn=self.mechanics.mean(1);can=(bn*w[:,0]+an*w[:,1])/(an+bn);crn=(w[:,0]-w[:,1])/(an+bn)
                    nominal=mn*float(can@operator(self.acc.t,self.windows,0,True)@self.acc.y)+In*float(crn@operator(self.yaw.t,self.windows,0,False)@self.yaw.y)
                    comp={'clock_signed':deterministic-zero,'nodes':m*float(abs(acc_cv)@self.acc.eps)+I*float(abs(yaw_cv)@self.yaw.eps),'bias':m*self.acc.bias*abs(acc_cv.sum())+I*self.yaw.bias*abs(yaw_cv.sum()),'motion_a':m*float(abs(KA).sum())*self.budget['rho_a_mps2'],'motion_r':I*float(abs(KD).sum())*self.budget['rho_r_rads'],'E':float(abs(KA)@self.E),'N':float(abs(KN)@self.N),'reference_at_witness_mechanics':zero}
                    comp['mechanics_signed']=comp.pop('reference_at_witness_mechanics')-nominal;comp['nominal_reference']=nominal
                    assert abs(sum(comp.values())-float(np.sum(w*response)))<1e-6
                    best={'value':value,'response':response,'mechanics':[m,I,a,b],'clocks':[rv['clock'],av['clock']],'components':comp,'fields':{'a':ea,'r':er,'E':ef,'N':en},'edges':self.edges,'endpoint_times':self.endpoint_times,'scope':'PL nodes plus common bounded measurable fields, not a tire-law trajectory','numerical_status':'float64_checked','generic_gap':0 if solver=='exact' else m*ag['gap']+I*rg['gap']}
        return best

    def interval(self,w,solver='exact'):
        lo=self.support(-np.asarray(w),solver);hi=self.support(w,solver);return np.array([-lo['value'],hi['value']]),lo,hi

    def query(self,identifier,w,units,tolerance,physical='empirical_unverified'):
        out={'id':identifier,'windows_s':self.windows.tolist(),'weights':np.asarray(w).tolist() if np.all(np.isfinite(w)) else None,'units':units,'witness_scope':'stated shared-field response class','physical_applicability':physical}
        try:
            validate(tolerance)
            if tolerance<0 or units not in ['N','N s']:raise ValueError('Unsupported tolerance/units')
            with np.errstate(over='raise',invalid='raise',divide='raise'):iv,lo,hi=self.interval(w)
            validate(iv);radius=float(np.diff(iv)[0]/2)
            status='report' if radius+1e-5<=tolerance else ('class_obstruction' if radius-1e-5>tolerance else 'unknown_computation')
            out.update(interval=iv.tolist(),radius=radius,conditional_decision=status,midpoint=float(iv.mean()) if status=='report' else None,numerical_status='float64_checked_not_certified',witness_targets=[float(np.sum(np.asarray(w)*lo['response'])),float(np.sum(np.asarray(w)*hi['response']))])
        except (ValueError,FloatingPointError,AssertionError) as e:out.update(conditional_decision='unknown',numerical_status='numerical_unknown',reason=str(e),interval=None,midpoint=None)
        return out

def catalogue(inputs,candidates):
    """Each unsupported candidate is marked without cancelling valid neighbors."""
    result=[]
    for c in candidates:
        try:
            M=SharedFields(windows=c['windows_s'],**inputs);result.append(M.query(c['id'],c['weights'],c['units'],c['tolerance']))
        except (ValueError,FloatingPointError) as e:
            result.append({'id':c['id'],'windows_s':c['windows_s'],'weights':c['weights'],'units':c['units'],'conditional_decision':'unknown','physical_applicability':'unverified','numerical_status':'unsupported_input','reason':str(e),'interval':None,'midpoint':None})
    return result
