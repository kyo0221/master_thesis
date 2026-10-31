import pickle,glob,numpy as np,bisect,math,sys
L=0.733
def seg_xy(D):
    tf=np.array([(r[1],r[2],r[3],r[5]) for r in D['tf:odom->base_link']]); tf=tf[np.argsort(tf[:,0])]
    od=np.array([(r[0],r[5]) for r in D['/localization/odom']])
    sp=D['/planner/speed_path']; spt=[r[0] for r in sp]
    out=[]
    for t,st,v in D['/cmd_vel']:
        if v<1.0: continue
        j=bisect.bisect_right(spt,t)-1
        if j<0 or t-spt[j]>0.3: continue
        P=sp[j][3]; i=min(max(bisect.bisect_left(tf[:,0],t),1),len(tf)-1); p=tf[i]
        c,s=math.cos(p[3]),math.sin(p[3]); dx=P[:,0]-p[1]; dy=P[:,1]-p[2]
        bx=c*dx+s*dy; by=-s*dx+c*dy
        k=min(bisect.bisect_left(od[:,0],t),len(od)-1)
        out.append((t,st,v,od[k,1],bx,by))
    return out
def pp(rows,lg,lmin,lmax):
    r=[]
    for t,st,v,vm,bx,by in rows:
        ld=np.clip(lg*vm,lmin,lmax); m=bx>0
        x,y=bx[m],by[m]
        if len(x)==0: r.append(0);continue
        d=np.hypot(x,y); k=np.argmax(d>=ld) if (d>=ld).any() else len(x)-1
        a=math.atan2(y[k],x[k]); r.append(math.atan2(2*L*math.sin(a),ld))
    return np.array(r)
if __name__=="__main__":
  for f in sys.argv[1:]:
     D=pickle.load(open(f,'rb')); rows=seg_xy(D)
     if len(rows)<100: print(f,'few');continue
     st=np.array([r[1] for r in rows]); m=np.abs(st)<0.25
     best=[]
     for lg in [0.6,1.0,1.2,1.5]:
       for lmin,lmax in [(3,5),(2,6),(3,8),(4,10),(5,10)]:
         x=pp(rows,lg,lmin,lmax)
         for lag in [0,1,2,3,5]:
             xs=x[:len(x)-lag] if lag else x; ys=st[lag:]; mm=m[lag:]
             g=(xs[mm]@ys[mm])/(xs[mm]@xs[mm]); c=np.corrcoef(xs[mm],ys[mm])[0,1]
             best.append((c,g,lg,lmin,lmax,lag))
     best.sort(reverse=True)
     print(f[:-4],' '.join(f"r={b[0]:.2f} g={b[1]:.2f} lg={b[2]} ld=[{b[3]},{b[4]}] lag={b[5]}" for b in best[:3]))
 