import pickle,glob,numpy as np,bisect,math
def modes(D):
    jt=[r[0] for r in D['/joy']]; J=D['/joy']; out=[]
    for t,st,v in D['/cmd_vel']:
        i=bisect.bisect_right(jt,t+0.003)-1; man=False
        if i>=0 and t-jt[i]<0.02:
            ax=J[i][1]; man=any(abs(st-0.2618*a)<1e-3 for a in ax) and any(abs(v-c*a)<1e-3 for a in ax for c in (3.0,4.0,5.0,5.5,1.0,2.0))
        out.append((t,st,v,man))
    return np.array(out)
if __name__=='__main__':
  for f in sorted(glob.glob('*.pkl')):
    D=pickle.load(open(f,'rb')); M=modes(D); t0=M[0,0]
    auto=(M[:,3]==0)&(M[:,2]>0.5)
    # segments
    segs=[];s=None
    for k in range(len(M)):
        if auto[k] and s is None: s=k
        if (not auto[k] or k==len(M)-1) and s is not None:
            if M[k-1,0]-M[s,0]>2: segs.append((round(M[s,0]-t0,1),round(M[k-1,0]-t0,1)))
            s=None
    # merge segments with gaps<1s
    mg=[]
    for a,b in segs:
        if mg and a-mg[-1][1]<1.5: mg[-1]=(mg[-1][0],b)
        else: mg.append((a,b))
    print(f"{f[:-4]:22s} manual={int(M[:,3].sum())} auto={int(auto.sum())} segs={mg}")
