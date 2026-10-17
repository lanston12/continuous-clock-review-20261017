"""Independent dense construction from native timestamps and PL hat primitives.

Preparation never instantiates SparseClockOracle or calls sparse preparation.
Only the compiled directional search kernel is shared, by design. Full-node
witnesses are decoded with the independent operator below for both methods.
"""
import numpy as np
from numba import njit
from sparse_clock import _maximize

@njit(cache=True)
def _raw_hat(t,i,z):
    if i>0 and t[i-1]<=z<=t[i]: return (z-t[i-1])/(t[i]-t[i-1])
    if i<len(t)-1 and t[i]<=z<=t[i+1]: return (t[i+1]-z)/(t[i+1]-t[i])
    return 0.

@njit(cache=True)
def _raw_slope(t,i,z):
    if i>0 and t[i-1]<z<t[i]: return 1./(t[i]-t[i-1])
    if i<len(t)-1 and t[i]<z<t[i+1]: return -1./(t[i+1]-t[i])
    return 0.

@njit(cache=True)
def _raw_primitive(t,i,z):
    result=0.
    if i>0:
        a,b=t[i-1],t[i]
        if z>=b: result+=(b-a)/2
        elif z>a: result+=(z-a)**2/(2*(b-a))
    if i<len(t)-1:
        a,b=t[i],t[i+1]
        if z>=b: result+=(b-a)/2
        elif z>a:
            v=z-a;result+=v-v*v/(2*(b-a))
    return result

@njit(cache=True)
def raw_operator(t,W,d,average):
    L=np.zeros((len(W),len(t)))
    for j in range(len(W)):
        s,e=W[j];h=e-s
        for i in range(len(t)):
            if average: L[j,i]=(_raw_primitive(t,i,e+d)-_raw_primitive(t,i,s+d))/h
            else: L[j,i]=(_raw_hat(t,i,e+d)-_raw_hat(t,i,s+d))/h
    return L

@njit(cache=True)
def prepare_dense_raw(t,y,W,edges,eps,average):
    K=len(edges)-1;J=len(W);n=len(t)
    Q=np.zeros((K,J,3));C=np.zeros((K,J,n,3))
    E=np.empty((K,n));N=np.full(K,n,np.int64)
    for k in range(K):
        lo,hi=edges[k],edges[k+1];mid=(lo+hi)/2;span=hi-lo
        for i in range(n): E[k,i]=eps[i]
        for j in range(J):
            s,e=W[j];h=e-s
            for i in range(n):
                if average:
                    c0=(_raw_primitive(t,i,e+lo)-_raw_primitive(t,i,s+lo))/h
                    c1=span*(_raw_hat(t,i,e+lo)-_raw_hat(t,i,s+lo))/h
                    c2=span*span*(_raw_slope(t,i,e+mid)-_raw_slope(t,i,s+mid))/(2*h)
                else:
                    c0=(_raw_hat(t,i,e+lo)-_raw_hat(t,i,s+lo))/h
                    c1=span*(_raw_slope(t,i,e+mid)-_raw_slope(t,i,s+mid))/h
                    c2=0.
                C[k,j,i,0]=c0;C[k,j,i,1]=c1;C[k,j,i,2]=c2
                Q[k,j,0]+=y[i]*c0;Q[k,j,1]+=y[i]*c1;Q[k,j,2]+=y[i]*c2
    return Q,C,E,N,np.zeros((K,1,J)),np.zeros((K,1)),np.zeros(K,np.int64)

class RawDenseClockOracle:
    def __init__(self,t,y,windows,clock,eps,average=True,bias=0.):
        t=np.asarray(t,float);y=np.asarray(y,float);W=np.asarray(windows,float);clock=np.asarray(clock,float)
        if any(not np.all(np.isfinite(x)) for x in [t,y,W,clock,eps,bias]): raise ValueError('Finite inputs required')
        if t.ndim!=1 or len(t)<2 or y.shape!=t.shape or np.any(np.diff(t)<=0): raise ValueError('Strict native nodes required')
        if W.ndim!=2 or W.shape[1]!=2 or len(W)==0 or np.any(W[:,1]<=W[:,0]): raise ValueError('Positive windows required')
        if clock.shape!=(2,) or clock[0]>clock[1]: raise ValueError('Ordered clock interval required')
        if W[:,0].min()+clock[0]<t[0] or W[:,1].max()+clock[1]>t[-1]: raise ValueError('No extrapolation')
        self.t=t;self.y=y;self.windows=W;self.clock=clock;self.average=bool(average);self.bias=float(bias)
        self.eps=np.broadcast_to(np.asarray(eps,float),y.shape).copy()
        if np.any(self.eps<0) or self.bias<0: raise ValueError('Nonnegative uncertainty required')
        crossing=(t[:,None]-W.ravel()[None,:]).ravel()
        self.edges=np.unique(np.r_[clock,crossing[(crossing>clock[0])&(crossing<clock[1])]])
        if len(self.edges)==1:self.edges=np.r_[self.edges,self.edges]
        self.prepared=prepare_dense_raw(t,y,W,self.edges,self.eps,self.average)
        self._cache={}
    def core(self,w):
        return _maximize(*self.prepared,self.edges,np.asarray(w,float),self.bias,self.average)
    @property
    def cells(self):
        return [(self.edges[k],self.edges[k+1],self.prepared[1][k]) for k in range(len(self.edges)-1)]
    def maximize(self,weights):
        w=np.asarray(weights,float)
        if w.shape!=(len(self.windows),) or not np.all(np.isfinite(w)):raise ValueError('Finite direction required')
        scale=float(np.max(abs(w)));w=w/scale if scale else w;key=tuple(w)
        if key not in self._cache:
            value,d,k,count=self.core(w)
            op=raw_operator(self.t,self.windows,d,self.average);c=w@op
            err=np.sign(c)*self.eps;bias=self.bias*np.sign(c.sum()) if self.average else 0.
            values=op@(self.y+err+bias);gap=float(value-w@values)
            if abs(gap)>1e-7*(1+abs(value)):raise FloatingPointError('Dense support/witness disagreement')
            self._cache[key]=dict(value=float(value),clock=float(d),values=values,node_errors=err,
                bias=float(bias),witness_gap=gap,knot_cells=len(self.edges)-1,sign_candidates=int(count))
        result=self._cache[key];return {**result,'value':result['value']*scale}
    def storage_bytes(self):
        return sum(x.nbytes for x in self.prepared)+self.edges.nbytes+self.t.nbytes+self.y.nbytes+self.eps.nbytes
