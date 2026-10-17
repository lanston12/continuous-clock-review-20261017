"""Endpoint-sparse continuous-clock support, with exact real-arithmetic reduction.

On a clock cell, only the hat nodes adjacent to 2J shifted window endpoints
can have nonconstant coefficients (at most 4J). Interior node uncertainty is
summed using positive hat-area prefix sums, not individually propagated.
Float64 root calculation is independently tested, not outward-rounded.
"""
import numpy as np
from numba import njit
from shared_node_support import operator

@njit(cache=True)
def _hat(t,i,z):
    n=len(t)
    if i>0 and t[i-1]<=z<=t[i]:return (z-t[i-1])/(t[i]-t[i-1])
    if i<n-1 and t[i]<=z<=t[i+1]:return (t[i+1]-z)/(t[i+1]-t[i])
    return 0.

@njit(cache=True)
def _hat_slope(t,i,z):
    if i>0 and t[i-1]<z<t[i]:return 1./(t[i]-t[i-1])
    if i<len(t)-1 and t[i]<z<t[i+1]:return -1./(t[i+1]-t[i])
    return 0.

@njit(cache=True)
def _hat_primitive(t,i,z):
    n=len(t);ans=0.
    if i>0:
        a=t[i-1];b=t[i]
        if z>=b:ans+=(b-a)/2.
        elif z>a:ans+=(z-a)**2/(2.*(b-a))
    if i<n-1:
        a=t[i];b=t[i+1]
        if z>=b:ans+=(b-a)/2.
        elif z>a:
            d=z-a;ans+=d-d*d/(2.*(b-a))
    return ans

@njit(cache=True)
def _signal_value(t,y,z):
    k=min(max(np.searchsorted(t,z,side='right')-1,0),len(t)-2)
    return y[k]+(y[k+1]-y[k])*(z-t[k])/(t[k+1]-t[k])

@njit(cache=True)
def _signal_slope(t,y,z):
    k=min(max(np.searchsorted(t,z,side='right')-1,0),len(t)-2)
    return (y[k+1]-y[k])/(t[k+1]-t[k])

@njit(cache=True)
def _signal_primitive(t,y,prefix,z):
    k=min(max(np.searchsorted(t,z,side='right')-1,0),len(t)-2);v=z-t[k]
    return prefix[k]+y[k]*v+.5*(y[k+1]-y[k])/(t[k+1]-t[k])*v*v

@njit(cache=True)
def _prepare(t,y,windows,edges,eps,average):
    n=len(t);J=len(windows);K=len(edges)-1;maxactive=4*J
    Q=np.zeros((K,J,3));C=np.zeros((K,J,maxactive,3));AE=np.zeros((K,maxactive));AN=np.zeros(K,np.int64)
    S=np.zeros((K,2*J-1,J));B=np.zeros((K,2*J-1));SN=np.zeros(K,np.int64)
    area=np.zeros(n);dt=np.diff(t);area[:-1]+=dt/2;area[1:]+=dt/2
    ep=np.zeros(n+1);ep[1:]=np.cumsum(eps*area)
    yp=np.zeros(n);yp[1:]=np.cumsum((y[1:]+y[:-1])*dt/2)
    wends=np.sort(np.unique(windows.ravel()))
    for k in range(K):
        lo=edges[k];hi=edges[k+1];mid=(lo+hi)/2;span=hi-lo
        active0=np.empty(4*J,np.int64);n0=0
        for j in range(J):
            for jj in range(2):
                z=windows[j,jj]+mid;ix=min(max(np.searchsorted(t,z,side='right')-1,0),n-2)
                active0[n0]=ix;active0[n0+1]=ix+1;n0+=2
        active=np.unique(active0[:n0]);AN[k]=len(active)
        for ii in range(len(active)):AE[k,ii]=eps[active[ii]]
        for j in range(J):
            s,e=windows[j];h=e-s;sl=s+lo;el=e+lo;sm=s+mid;em=e+mid
            if average:
                Q[k,j,0]=(_signal_primitive(t,y,yp,el)-_signal_primitive(t,y,yp,sl))/h
                Q[k,j,1]=span*(_signal_value(t,y,el)-_signal_value(t,y,sl))/h
                Q[k,j,2]=span*span*(_signal_slope(t,y,em)-_signal_slope(t,y,sm))/(2.*h)
            else:
                Q[k,j,0]=(_signal_value(t,y,el)-_signal_value(t,y,sl))/h
                Q[k,j,1]=span*(_signal_slope(t,y,em)-_signal_slope(t,y,sm))/h
            for ii in range(len(active)):
                idx=active[ii]
                if average:
                    C[k,j,ii,0]=(_hat_primitive(t,idx,el)-_hat_primitive(t,idx,sl))/h
                    C[k,j,ii,1]=span*(_hat(t,idx,el)-_hat(t,idx,sl))/h
                    C[k,j,ii,2]=span*span*(_hat_slope(t,idx,em)-_hat_slope(t,idx,sm))/(2.*h)
                else:
                    C[k,j,ii,0]=(_hat(t,idx,el)-_hat(t,idx,sl))/h
                    C[k,j,ii,1]=span*(_hat_slope(t,idx,em)-_hat_slope(t,idx,sm))/h
        if average:
            for u in range(len(wends)-1):
                left=wends[u]+mid;right=wends[u+1]+mid;wm=(wends[u]+wends[u+1])/2
                li=np.searchsorted(t,left,side='right');ri=np.searchsorted(t,right,side='left')
                mass=ep[ri]-ep[li]
                for ix in active:
                    if li<=ix<ri:mass-=eps[ix]*area[ix]
                # Recompute cancellation-dominated positive sums; do not clamp.
                if mass<=64*2.220446049250313e-16*max(ep[ri],ep[li]):
                    mass=0.
                    for ix in range(li,ri):
                        is_active=False
                        for aidx in active:
                            if ix==aidx:is_active=True;break
                        if not is_active:mass+=eps[ix]*area[ix]
                if mass<0 or not np.isfinite(mass):
                    raise FloatingPointError('Unresolved group budget')
                B[k,u]=mass
                for j in range(J):
                    if windows[j,0]<wm<windows[j,1]:S[k,u,j]=1./(windows[j,1]-windows[j,0])
            SN[k]=len(wends)-1
    return Q,C,AE,AN,S,B,SN

@njit(cache=True)
def _roots_inside(c0,c1,c2,out,n):
    scale=max(abs(c0),abs(c1),abs(c2))
    if scale==0:return n
    z=c0/scale;b=c1/scale;a=c2/scale
    if a==0:
        if b!=0:
            x=-z/b
            if 0<x<1:out[n]=x;n+=1
        return n
    disc=b*b-4*a*z
    if disc<0:return n
    q=-.5*(b+np.copysign(np.sqrt(disc),b))
    if q==0:
        x=-b/(2*a)
        if 0<x<1:out[n]=x;n+=1
    else:
        x=q/a
        if 0<x<1:out[n]=x;n+=1
        x=z/q
        if 0<x<1:out[n]=x;n+=1
    return n

@njit(cache=True)
def _f(x,q,c,e,na,const):
    val=q[0]+q[1]*x+q[2]*x*x+const
    for i in range(na):val+=e[i]*abs(c[i,0]+c[i,1]*x+c[i,2]*x*x)
    return val

@njit(cache=True)
def _maximize(Q,C,AE,AN,S,B,SN,edges,w,bias,average):
    best=-np.inf;bestclock=edges[0];bestcell=0;count=0
    K,J,_=Q.shape;amax=C.shape[2]
    for k in range(K):
        q=np.zeros(3);c=np.empty((amax,3));ce=np.empty(amax);na=0
        for j in range(J):
            q[0]+=w[j]*Q[k,j,0];q[1]+=w[j]*Q[k,j,1];q[2]+=w[j]*Q[k,j,2]
        const=bias*abs(w.sum()) if average else 0.
        for u in range(SN[k]):
            v=0.
            for j in range(J):v+=S[k,u,j]*w[j]
            const+=B[k,u]*abs(v)
        # Identical direction-level simplification for dense and compressed inputs.
        # Constant columns are summed once; no temporary three-element arrays.
        for i in range(AN[k]):
            p0=0.;p1=0.;p2=0.
            for j in range(J):
                p0+=w[j]*C[k,j,i,0];p1+=w[j]*C[k,j,i,1];p2+=w[j]*C[k,j,i,2]
            if p1==0. and p2==0.:
                const+=AE[k,i]*abs(p0)
            else:
                c[na,0]=p0;c[na,1]=p1;c[na,2]=p2;ce[na]=AE[k,i];na+=1
        roots=np.empty(2*amax+2);roots[0]=0.;roots[1]=1.;nn=2
        for i in range(na):nn=_roots_inside(c[i,0],c[i,1],c[i,2],roots,nn)
        cuts=np.sort(roots[:nn]);count+=nn
        for x in cuts:
            value=_f(x,q,c,ce,na,const)
            if value>best:best=value;bestclock=edges[k]+(edges[k+1]-edges[k])*x;bestcell=k
        for kk in range(nn-1):
            left=cuts[kk];right=cuts[kk+1]
            if right<=left:continue
            mid=(left+right)/2;p1=q[1];p2=q[2]
            for i in range(na):
                sg=ce[i]*np.sign(c[i,0]+c[i,1]*mid+c[i,2]*mid*mid)
                p1+=sg*c[i,1];p2+=sg*c[i,2]
            if p2<0:
                x=-p1/(2*p2)
                if left<x<right:
                    value=_f(x,q,c,ce,na,const);count+=1
                    if value>best:best=value;bestclock=edges[k]+(edges[k+1]-edges[k])*x;bestcell=k
    return best,bestclock,bestcell,count

class SparseClockOracle:
    def __init__(self,t,y,windows,clock,eps,average=True,bias=0.):
        t=np.asarray(t,float);y=np.asarray(y,float);windows=np.asarray(windows,float);clock=np.asarray(clock,float)
        if any(not np.all(np.isfinite(z)) for z in [t,y,windows,clock,eps,bias]):raise ValueError('Finite inputs required')
        if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Strict sensor nodes required')
        if windows.ndim!=2 or windows.shape[1]!=2 or len(windows)==0 or np.any(windows[:,1]<=windows[:,0]):raise ValueError('Positive windows required')
        if clock.shape!=(2,) or clock[1]<clock[0]:raise ValueError('Ordered clock bounds required')
        if windows[:,0].min()+clock[0]<t[0]-1e-12 or windows[:,1].max()+clock[1]>t[-1]+1e-12:raise ValueError('No extrapolation')
        self.t,self.y,self.windows,self.clock,self.average=t,y,windows,clock,average
        self.eps=np.broadcast_to(np.asarray(eps,float),y.shape).copy();self.bias=float(bias)
        if np.any(self.eps<0) or bias<0:raise ValueError('Nonnegative uncertainty')
        edges=np.unique(np.r_[clock,np.concatenate([t[(t-z>clock[0])&(t-z<clock[1])]-z for z in windows.ravel()])])
        if len(edges)==1:edges=np.r_[edges,edges]
        self.edges=edges;self.prepared=_prepare(t,y,windows,edges,self.eps,average);self._cache={}
    def maximize(self,weights):
        weights=np.asarray(weights,float)
        if weights.shape!=(len(self.windows),) or not np.all(np.isfinite(weights)):raise ValueError('Finite directional weights required')
        scale=float(np.max(abs(weights)));w=weights/scale if scale else weights;key=tuple(w)
        if key not in self._cache:
            value,d,k,count=_maximize(*self.prepared,self.edges,w,self.bias,self.average)
            # Independent dense-operator witness evaluation; not used in support search.
            op=operator(self.t,self.windows,d,self.average);cv=w@op;err=np.sign(cv)*self.eps;bias=np.sign(cv.sum())*self.bias
            values=op@(self.y+err+bias);witness=float(w@values)
            if abs(value-witness)>1e-7*(1+abs(value)):raise FloatingPointError(f'Sparse witness gap {value-witness}')
            self._cache[key]={'value':float(value),'clock':float(d),'node_errors':err,'bias':float(bias),'values':values,'knot_cells':len(self.edges)-1,'sign_candidates':int(count),'active_nodes_max':int(max(self.prepared[3])),'witness_gap':float(value-witness)}
        out=self._cache[key];return {**out,'value':out['value']*scale}
    def storage_bytes(self):return sum(x.nbytes for x in self.prepared)+self.edges.nbytes+self.t.nbytes+self.y.nbytes+self.eps.nbytes
