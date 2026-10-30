"""vision_lane_planner 後処理 (lane_path_builder.cpp) の Python 移植"""
import numpy as np
FX=FY=244.3825225830078; CX=324.4415588378906; CY=175.32806396484375; W,H=640,360
ROW_START=25; NR=23; PADTOP=12; PADLEFT=0; IW=640; IH=384; RAC=48
MAXR=15.0; MAXGAP=10.0
def rot_rpy(r,p,y):
    r,p,y=np.radians([r,p,y]); cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]);Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]);Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx
R=rot_rpy(-89.801314,0.497098,-89.640661); T=np.array([-0.008,0.0325,0.5547])
def row_v(row): rh=IH/RAC; return (ROW_START+row)*rh+(rh-1)/2
def project(pos,valid_logit,thr=0.5):
    """pos,valid_logit: (23,) straight スロット。戻り: 近い順 (M,2) と (u,v)"""
    pts=[];uv=[]
    for off in range(NR):
        row=NR-1-off
        if 1/(1+np.exp(-float(valid_logit[row])))<=thr: continue
        u=float(pos[row])*(IW-1)-PADLEFT; v=row_v(row)-PADTOP
        if u<0 or u>W-1 or v<0 or v>H-1: continue
        cv=np.array([(u-CX)/FX,(v-CY)/FY,1.0]); bv=R@cv; sc=(0-T[2])/bv[2]
        if not np.isfinite(sc) or sc<=0: continue
        p=R@(sc*cv)+T; pts.append(p[:2]); uv.append((u,v))
    return np.array(pts).reshape(-1,2),uv
def truncate(P):
    kept=[]
    for p in P:
        if p[0]>MAXR: break
        if not kept:
            if p[0]<=0: continue
        else:
            if p[0]<=kept[-1][0]: break
            if np.hypot(*(p-kept[-1]))>MAXGAP: break
        kept.append(p)
    return np.array(kept).reshape(-1,2)
def build(pos,vl):
    """base_link Nx3 [x,y,yaw] (N<2 なら空)"""
    s=truncate(project(pos,vl)[0])
    if len(s)<2: return np.zeros((0,3))
    yaw=np.zeros(len(s)); yaw[:-1]=np.arctan2(np.diff(s[:,1]),np.diff(s[:,0])); yaw[-1]=yaw[-2]
    return np.c_[s,yaw]
