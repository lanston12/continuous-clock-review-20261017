"""Independent four-wheel lateral/roll vehicle generator (SI, x forward, y left).

This is an explicitly specified research simulation, not CarSim or real-vehicle data.
Nonlinear load-sensitive friction saturation and first-order force relaxation are
used to generate truth. Reconstruction does not receive these constitutive inputs.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

@njit(cache=True)
def control(t, mode, amp, phase):
    # A smooth one-second start avoids an artificial input discontinuity.
    start=min(max((t-.5)/1.,0.),1.)
    gate=start*start*(3.-2.*start)
    if mode == 0: # multisine
        u=amp*(.72*np.sin(2*np.pi*.34*t+phase)+.28*np.sin(2*np.pi*1.03*t+.4))
    elif mode == 1: # chirp
        u=amp*np.sin(2*np.pi*(.18*t+.045*t*t)+phase)
    elif mode == 2: # smooth two-lobe steering pulse
        u=amp*(np.exp(-((t-3.2)/.70)**2)-np.exp(-((t-5.3)/.70)**2))
    else: # modulated nonlinear steer
        u=amp*np.sin(2*np.pi*.42*t+phase)*(.7+.3*np.sin(2*np.pi*.075*t))
    return gate*u

@njit(cache=True)
def rhs(t,x,p,mode,law,amp,phase):
    # p: m, Iz, a,b, track, u, mu, Cwheel, sigma, Isroll, Ixz,Kroll,Droll,h_s,m_s
    m,Iz,a,b,track,u,mu,C0,sigma,Jroll,Ixz,Kroll,Droll,h,ms=p
    vy,r,phi,pr=x[:4]
    delta=control(t,mode,amp,phase)
    # Roll-supported lateral transfer, split between axles; clamp is a specified
    # unloading approximation, not a claim to tire/contact fidelity beyond this model.
    roll_moment=Kroll*phi+Droll*pr
    tf=.57*roll_moment/track;tr=.43*roll_moment/track
    fzf=m*9.80665*b/(a+b)/2.;fzr=m*9.80665*a/(a+b)/2.
    loads=np.array([max(.18*fzf,fzf-tf),max(.18*fzf,fzf+tf),max(.18*fzr,fzr-tr),max(.18*fzr,fzr+tr)])
    xx=np.array([a,a,-b,-b]);yy=np.array([track/2,-track/2,track/2,-track/2])
    # Moderate, time-varying longitudinal force imbalance supplies a genuine
    # omitted track-width yaw moment. Axle force targets include steer projection.
    imbalance=65.*np.sin(1.15*t+phase)
    Fx=np.array([85.+imbalance,85.-imbalance,55.-.7*imbalance,55.+.7*imbalance])
    B=np.zeros(4);Xb=np.zeros(4);dFy=np.zeros(4)
    for k in range(4):
        de=delta if k<2 else 0.
        localu=max(3.,u-yy[k]*r)
        slip=de-np.arctan2(vy+xx[k]*r,localu)
        cap=np.sqrt(max(1.,(mu*loads[k])**2-Fx[k]**2))
        C=C0*(loads[k]/(m*9.80665/4.))**.85
        z=C*slip/cap
        if law==0: steady=cap*np.tanh(z)
        else: steady=cap*(2./np.pi)*np.arctan(np.pi*z/2.)
        dFy[k]=(steady-x[4+k])*localu/(sigma*(1.+.12*k))
        B[k]=x[4+k]*np.cos(de)+Fx[k]*np.sin(de)
        Xb[k]=Fx[k]*np.cos(de)-x[4+k]*np.sin(de)
    E=30.*np.sin(.61*t+phase)+15.*np.sin(1.7*t)
    externalM=35.*np.sin(.73*t+.3)
    Ay=(B.sum()+E)/m
    direct=a*(B[0]+B[1])-b*(B[2]+B[3])
    moment=direct-np.dot(yy,Xb)+externalM
    roll_rhs=ms*h*Ay-Kroll*phi-Droll*pr
    det=Iz*Jroll-Ixz*Ixz
    rd=(Jroll*moment-Ixz*roll_rhs)/det
    prd=(Iz*roll_rhs-Ixz*moment)/det
    dx=np.empty(8);dx[0]=Ay-u*r;dx[1]=rd;dx[2]=pr;dx[3]=prd;dx[4:]=dFy
    # Balance residues are computed for evaluating/calibrating the model class.
    N=Iz*rd-direct
    return dx,Ay,B,E,N,delta

@njit(cache=True)
def simulate(p,mode,law,amp,phase,dt=.002,end=10.):
    t=np.arange(0.,end+dt*.1,dt);n=len(t);x=np.zeros((n,8));A=np.zeros(n);B=np.zeros((n,4));E=np.zeros(n);N=np.zeros(n);steer=np.zeros(n)
    for k in range(n-1):
        k1=rhs(t[k],x[k],p,mode,law,amp,phase)[0]
        k2=rhs(t[k]+dt/2,x[k]+dt*k1/2,p,mode,law,amp,phase)[0]
        k3=rhs(t[k]+dt/2,x[k]+dt*k2/2,p,mode,law,amp,phase)[0]
        k4=rhs(t[k]+dt,x[k]+dt*k3,p,mode,law,amp,phase)[0]
        x[k+1]=x[k]+dt*(k1+2*k2+2*k3+k4)/6
    for k in range(n):
        _,A[k],B[k],E[k],N[k],steer[k]=rhs(t[k],x[k],p,mode,law,amp,phase)
    return t,x,A,B,E,N,steer

def specification(seed, mode, law):
    g=np.random.default_rng(seed)
    m=g.uniform(1250,2150);a=g.uniform(1.03,1.38);b=g.uniform(1.22,1.67)
    Iz=m*g.uniform(1.32,1.85);track=g.uniform(1.48,1.70);u=g.uniform(12.,25.);mu=g.uniform(.55,1.0)
    C=g.uniform(34000,52000);sigma=g.uniform(.55,1.35);Jroll=g.uniform(430,760);Ixz=g.uniform(-65,65)
    K=g.uniform(60000,85000);D=g.uniform(4500,6500);hs=g.uniform(.40,.60);ms=.87*m
    amp=g.uniform(.025,.075);phase=g.uniform(-np.pi,np.pi)
    return {'seed':int(seed),'mode':int(mode),'law':int(law),'parameters':[m,Iz,a,b,track,u,mu,C,sigma,Jroll,Ixz,K,D,hs,ms],'amp':float(amp),'phase':float(phase),'generator':'four-wheel roll-coupled ODE, not CarSim'}

def generate(spec,dt=.002):
    t,x,A,B,E,N,steer=simulate(np.array(spec['parameters']),spec['mode'],spec['law'],spec['amp'],spec['phase'],dt)
    p=spec['parameters'];out={'t':t,'r':x[:,1],'vy':x[:,0],'roll':x[:,2],'roll_rate':x[:,3],'A':A,'F':np.column_stack([B[:,:2].sum(1),B[:,2:].sum(1)]),'E':E,'N':N,'steer':steer,'mechanics':np.array(p[:4]),'speed':np.full(len(t),p[5])}
    if not all(np.all(np.isfinite(v)) for v in out.values()):raise FloatingPointError('Nonfinite simulation')
    return out
