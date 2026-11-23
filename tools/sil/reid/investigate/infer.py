import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
RESD=ROOT+"/out_reid3/res"
KP,KI,LPF,DT=1.5,0.3,0.5,0.05
def infer(tag,t0,tend=None):
    """creep 区間の a_cmd = KP*fe + KI*I (a_ref=0) から I を求め、I' = fe を逆向きに積分して接近区間の I と a_ref を復元"""
    L,A=loadr(tag,RESD); pl=A['plan']; t=L[:,0]; vm=L[:,3]; vc=L[:,6]
    vref=pl[:,2].copy()
    for i in range(1,len(vref)):
        if not np.isfinite(vref[i]): vref[i]=vref[i-1]
    e=vref-vm; fe=np.zeros_like(e)
    i0=np.searchsorted(t,t0-6)
    fe[i0]=e[i0]
    for i in range(i0+1,len(e)): fe[i]=LPF*fe[i-1]+(1-LPF)*e[i]
    acmd=np.zeros_like(vc); acmd[:-1]=np.diff(vc)/DT
    # creep: vref==0 and vm>0.08 first run after t0
    ic=np.where((t>t0)&(vref<0.01)&(vm>0.1)&(vm<0.5))[0]
    return L,t,vm,vc,vref,fe,acmd,ic
if __name__=='__main__':
    tag=sys.argv[1]; t0=float(sys.argv[2])
    L,t,vm,vc,vref,fe,acmd,ic=infer(tag,t0)
    print('creep idx',ic[0],ic[-1],'t-t0',t[ic[0]]-t0,t[ic[-1]]-t0)
    # I estimate from each creep sample
    Iest=(acmd[ic]-KP*fe[ic])/KI
    print('I est first/mid/last',Iest[:3],Iest[len(Iest)//2],Iest[-3:])
    # forward consistency: I[i+1]=I[i]+fe*DT starting from I at ic[5]
    j=ic[5]; I=(np.mean(acmd[j-2:j+3])-KP*np.mean(fe[j-2:j+3]))/KI
    Ifw=[I]; 
    for i in range(j,ic[-1]): I+=fe[i]*DT; Ifw.append(I)
    Ifw=np.array(Ifw); Iobs=(acmd[j:ic[-1]+1]-KP*fe[j:ic[-1]+1])/KI
    print('fw vs obs rms',np.sqrt(np.mean((Ifw-Iobs)**2)),Ifw[::20],Iobs[::20])

def first_run(ic):
    br=np.where(np.diff(ic)>1)[0]
    return ic[:br[0]+1] if len(br) else ic
def Itrace(tag,t0):
    L,t,vm,vc,vref,fe,acmd,ic=infer(tag,t0); ic=first_run(ic)
    j=ic[3]; Ij=(np.mean(acmd[j-1:j+2])-KP*np.mean(fe[j-1:j+2]))/KI
    ia=np.searchsorted(t,t0-1.0)
    I=np.zeros(len(t)); I[j]=Ij
    for i in range(j,ia,-1): I[i-1]=I[i]-fe[i-1]*DT   # I[i]=I[i-1]+fe[i]*DT -> I[i-1]=I[i]-fe[i]*DT
    for i in range(j,ic[-1]): I[i+1]=I[i]+fe[i+1]*DT
    return dict(t=t,vm=vm,vc=vc,vref=vref,fe=fe,acmd=acmd,I=I,ia=ia,j=j,ic=ic,L=L)
