import pickle,numpy as np,math,bisect,re
from mode import modes
LAT0,LON0,YAW=36.11313110136178,139.98107529988687,0.02686650
R=6378137.0
def enu(lat,lon):
    e=np.radians(np.asarray(lon)-LON0)*R*math.cos(math.radians(LAT0)); n=np.radians(np.asarray(lat)-LAT0)*R
    return e,n
def to_map(e,n,yaw=YAW):
    c,s=math.cos(yaw),math.sin(yaw); return c*e+s*n, -s*e+c*n
def load_osm(path='/home/kyo/formula_ws/src/map/vectormap_server/config/aiformula_course.osm'):
    txt=open(path).read(); N={m[0]:(float(m[1]),float(m[2])) for m in re.findall(r'<node id="(\d+)"[^>]*x="([-\d.e]+)" y="([-\d.e]+)"',txt)}
    ways=[]
    for w in re.findall(r'<way.*?</way>',txt,re.S):
        refs=re.findall(r'<nd ref="(\d+)"',w); ty=re.findall(r'k="type" v="([^"]+)"',w)
        pts=np.array([N[r] for r in refs if r in N])
        if len(pts)>1: ways.append((ty[0] if ty else '',pts))
    return ways
def topo(path='/home/kyo/Downloads/aiformula_topomap.osm'):
    txt=open(path).read(); out={}
    for m in re.findall(r'<node id="(\d+)"[^>]*lat="([-\d.]+)" lon="([-\d.]+)"',txt):
        e,n=enu(float(m[1]),float(m[2])); out[m[0]]=to_map(e,n)
    return out
def arr(D,k,cols=None):
    return np.array([r[1:] if cols is None else [r[c] for c in cols] for r in D[k]],dtype=float)
def gnss_map(D):
    G=np.array([(r[0],r[2],r[3]) for r in D['/vectornav/gnss'] if abs(r[2])>1]); x,y=to_map(*enu(G[:,1],G[:,2])); return G[:,0],x,y
def interp(t,tt,v): return np.interp(t,tt,v)
def lateral_at0(D,key='/planner/global_path'):
    out=[]
    for r in D[key]:
        P=r[3]
        if len(P)<2: continue
        k=np.argsort(P[:,0]); 
        if P[k[0],0]>1.5: out.append((r[0],np.nan,np.nan)); continue
        y0=np.interp(0.0,P[k,0],P[k,1]); y5=np.interp(5.0,P[k,0],P[k,1]) if P[:,0].max()>5 else np.nan
        out.append((r[0],y0,y5))
    return np.array(out)
