"""Finite continuous-clock support of a shared-node PL balance response class.

No grid maximum is used as an outer bound. Float64 extrema are not directed-
rounding certificates. Attaining witnesses concern this class, not a tire law.
"""
import itertools
import numpy as np

def point_weights(t,z):
    t=np.asarray(t,float)
    if z<t[0]-1e-12 or z>t[-1]+1e-12:raise ValueError('No extrapolation')
    z=np.clip(z,t[0],t[-1]);k=min(max(np.searchsorted(t,z,side='right')-1,0),len(t)-2)
    f=(z-t[k])/(t[k+1]-t[k]);v=np.zeros(len(t));v[k]=1-f;v[k+1]=f;return v

def primitive_weights(t,z):
    if z<t[0]-1e-12 or z>t[-1]+1e-12:raise ValueError('No extrapolation')
    z=np.clip(z,t[0],t[-1]);k=min(max(np.searchsorted(t,z,side='right')-1,0),len(t)-2)
    dt=np.diff(t);v=np.zeros(len(t));v[:k]+=dt[:k]/2;v[1:k+1]+=dt[:k]/2
    d=z-t[k];v[k]+=d-d*d/(2*dt[k]);v[k+1]+=d*d/(2*dt[k]);return v

def operator(t,windows,d,average):
    rows=[]
    for s,e in np.asarray(windows):
        fun=primitive_weights if average else point_weights
        rows.append((fun(t,e+d)-fun(t,s+d))/(e-s))
    return np.asarray(rows)

def _real_roots(c):
    # Coefficients use x in [0,1], avoiding tiny clock-cell conditioning.
    c=np.asarray(c,np.longdouble);scale=max(np.max(abs(c)),np.longdouble('1e-300'));a,b,z=c[2]/scale,c[1]/scale,c[0]/scale
    if a==0:return [] if b==0 else [float(-z/b)]
    D=b*b-4*a*z
    if D<0:return []
    q=-(b+np.copysign(np.sqrt(D),b))/2
    if q==0:return [float(-b/(2*a))]
    return [float(q/a),float(z/q)]

class ClockOracle:
    def __init__(self,t,y,windows,clock,eps,average=True,bias=0.):
        t=np.asarray(t,float);y=np.asarray(y,float);windows=np.asarray(windows,float)
        if any(not np.all(np.isfinite(z)) for z in [t,y,windows,clock,eps,bias]):raise ValueError('Nonfinite input')
        if t.ndim!=1 or y.ndim!=1 or windows.ndim!=2 or windows.shape[1]!=2 or len(clock)!=2:raise ValueError('Input shape')
        if len(t)<2 or np.any(np.diff(t)<=0) or len(t)!=len(y):raise ValueError('Strict node clock')
        if np.any(windows[:,1]<=windows[:,0]) or clock[0]>clock[1]:raise ValueError('Window/clock interval')
        if windows[:,0].min()+clock[0]<t[0]-1e-12 or windows[:,1].max()+clock[1]>t[-1]+1e-12:raise ValueError('Support incomplete')
        self.t,self.y,self.windows,self.clock,self.average=t,y,windows,clock,average
        self.eps=np.broadcast_to(np.asarray(eps,float),y.shape).copy();self.bias=float(bias)
        if np.any(self.eps<0) or bias<0:raise ValueError('Positive error bounds')
        edges=np.unique(np.r_[clock,[v-z for z in windows.ravel() for v in t if clock[0]<v-z<clock[1]]])
        self.cells=[]
        if len(edges)==1:edges=np.r_[edges,edges]
        for lo,hi in zip(edges[:-1],edges[1:]):
            v0=operator(t,windows,lo,average);vm=operator(t,windows,(lo+hi)/2,average);v1=operator(t,windows,hi,average)
            c2=2*(v1+v0-2*vm);c1=v1-v0-c2
            self.cells.append((lo,hi,np.stack([v0,c1,c2],axis=-1)))

    def maximize(self,weights):
        weights=np.asarray(weights,float);best=None;candidate_count=0
        if weights.shape!=(len(self.windows),) or not np.all(np.isfinite(weights)):raise ValueError('Finite weight vector')
        for lo,hi,P in self.cells:
            c=np.einsum('j,jnp->np',weights,P);q=self.y@c
            roots=[0.,1.]
            for co in c:
                roots.extend(x for x in _real_roots(co) if 1e-12<x<1-1e-12)
            roots=np.unique(roots);candidate_count+=len(roots)
            def val(x):
                cv=c@np.array([1.,x,x*x]);return float(cv@self.y+abs(cv)@self.eps+self.bias*abs(cv.sum()))
            candidates=list(roots)
            for l,r in zip(roots[:-1],roots[1:]):
                x=(l+r)/2;sgn=np.sign(c@np.array([1.,x,x*x]));poly=q+(self.eps*sgn)@c
                # Shared bias is constant: averages sum weights; increments sum 0.
                if poly[2]<-1e-13:
                    stationary=-poly[1]/(2*poly[2])
                    if l<stationary<r:candidates.append(stationary)
            values=[val(x) for x in candidates];k=int(np.argmax(values));x=candidates[k];value=values[k]
            if best is None or value>best['value']:
                d=lo+(hi-lo)*x;cv=weights@operator(self.t,self.windows,d,self.average)
                err=np.sign(cv)*self.eps;bias=np.sign(cv.sum())*self.bias
                best={'value':value,'clock':float(d),'node_errors':err,'bias':float(bias),'values':operator(self.t,self.windows,d,self.average)@(self.y+err+bias)}
        best['knot_cells']=len(self.cells);best['sign_candidates']=candidate_count
        return best

class JointResponse:
    def __init__(self,tr,yr,ta,ya,windows,dr,da,mechanics,eps_r,eps_a,rho_r=0.,rho_a=0.,E=0.,N=0.,bias_r=0.,bias_a=0.):
        self.windows=np.asarray(windows,float);self.mechanics=np.asarray(mechanics,float)
        if self.mechanics.shape!=(4,2) or np.any(self.mechanics<=0) or np.any(np.diff(self.mechanics,axis=1)<0):raise ValueError('Positive mechanics box')
        self.yaw=ClockOracle(tr,yr,windows,dr,eps_r,False,bias_r);self.acc=ClockOracle(ta,ya,windows,da,eps_a,True,bias_a)
        h=np.diff(self.windows,axis=1)[:,0]
        self.guards=np.stack([np.broadcast_to(rho_a,h.shape),2*np.broadcast_to(rho_r,h.shape)/h,np.broadcast_to(E,h.shape),np.broadcast_to(N,h.shape)],axis=1)
        if np.any(self.guards<0):raise ValueError('Nonnegative residual guards')

    def support(self,w):
        w=np.asarray(w,float)
        if w.shape!=(len(self.windows),2):raise ValueError('Target weights')
        best=None
        # Positive m,I only scale the independent sensor maxima. Cache them
        # outside the m,I corner loop; still handles uncertain a,b.
        for a,b in set(itertools.product(*self.mechanics[2:])):
            L=a+b;cA=(b*w[:,0]+a*w[:,1])/L;cR=(w[:,0]-w[:,1])/L
            av=self.acc.maximize(cA);rv=self.yaw.maximize(cR)
            for m,I in set(itertools.product(*self.mechanics[:2])):
                coeff=np.c_[m*cA,I*cR,-cA,-cR];err=np.sign(coeff)*self.guards
                A=av['values']+err[:,0];D=rv['values']+err[:,1];F=m*A-err[:,2];Q=I*D-err[:,3]
                response=np.c_[(b*F+Q)/L,(a*F-Q)/L];value=float(m*av['value']+I*rv['value']+np.sum(abs(coeff)*self.guards))
                if best is None or value>best['value']:best={'value':value,'mechanics':[m,I,a,b],'clocks':[rv['clock'],av['clock']],'response':response,'yaw':rv,'acc':av,'residual_witness':err}
        return best

    def interval(self,w):
        lo=self.support(-np.asarray(w));hi=self.support(w)
        return np.array([-lo['value'],hi['value']]),lo,hi

def reporting(interval,tolerance,applicable=True,numeric_allowance=1e-7,witness_interval=None):
    """Class-specific status; no physical necessity claim for relaxed residuals.

    Numeric allowance is a declared empirical guard, not a certificate.
    A lower separation must come from evaluated admissible witness targets.
    """
    if tolerance<0:raise ValueError('Nonnegative tolerance')
    radius=(interval[1]-interval[0])/2+numeric_allowance
    if not applicable:return {'status':'unknown_assumptions','radius':float(radius),'estimate':None}
    if radius<=tolerance:return {'status':'report','radius':float(radius),'estimate':float(np.mean(interval))}
    if witness_interval is not None and (witness_interval[1]-witness_interval[0])/2-numeric_allowance>tolerance:
        return {'status':'class_obstruction','radius':float(radius),'estimate':None}
    return {'status':'unknown_computation','radius':float(radius),'estimate':None}
