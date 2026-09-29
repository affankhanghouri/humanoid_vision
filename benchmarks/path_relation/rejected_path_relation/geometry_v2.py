"""Occlusion-aware probes, affine alignment and conservative corridor geometry."""
from dataclasses import dataclass
import cv2
import numpy as np
from rejected_path_relation.geometry import CorridorGeometry

@dataclass(frozen=True)
class Probe:
    name: str
    center: tuple[float, float]
    radii: tuple[float, float]
    usable: bool

def object_probes(box, frame_size):
    x1,y1,x2,y2=box; fw,fh=frame_size; w,h=max(1.,x2-x1),max(1.,y2-y1); cx=(x1+x2)*.5
    specs=(("contact",cx,y2-.015*h,.10*w,.025*h),("below",cx,y2+.18*h,.18*w,.05*h),
           ("left",x1-.16*w,y2-.02*h,.075*w,.06*h),("right",x2+.16*w,y2-.02*h,.075*w,.06*h))
    return tuple(Probe(n,(float(np.clip(x,0,fw-1)),float(np.clip(y,0,fh-1))),
                 (max(1.,rx),max(1.,ry)),x-rx>=0 and x+rx<fw and y-ry>=0 and y+ry<fh)
                 for n,x,y,rx,ry in specs)

def compose_inverse(snapshot, source_timestamp, target_timestamp, max_age=1.05):
    """Map target-frame motion pixels back to source-frame motion pixels."""
    if snapshot is None or target_timestamp-source_timestamp < -1e-6 or target_timestamp-source_timestamp > max_age:
        return None
    extrapolate = target_timestamp - snapshot.timestamp
    if extrapolate > .20:return None
    chain_target = min(target_timestamp, snapshot.timestamp)
    matrix=np.eye(3); at=source_timestamp; used=0
    for step in snapshot.steps:
        if step.end <= at+1e-6:continue
        if step.start > at+1e-5 or step.affine is None or step.end > chain_target+1e-5:return None
        part=np.eye(3); part[:2]=step.affine; matrix=part@matrix; at=step.end; used+=1
        if at>=chain_target-1e-6:break
    if abs(at-chain_target)>1e-5 and abs(source_timestamp-chain_target)>1e-5:return None
    if not used and abs(source_timestamp-chain_target)>1e-5:return None
    if extrapolate > 1e-6:
        if not snapshot.steps or snapshot.steps[-1].affine is None:return None
        step=snapshot.steps[-1]; fraction=extrapolate/(step.end-step.start)
        part=np.eye(3); part[:2]=np.eye(2,3)+fraction*(step.affine-np.eye(2,3)); matrix=part@matrix
    try:return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:return None

def transform_point(point,inverse,scale):
    p=np.array((point[0]*scale[0],point[1]*scale[1],1.)); q=inverse@p
    return float(q[0]/scale[0]),float(q[1]/scale[1])

def construct_forward_corridor_v2(drivable_mask,content_rect):
    mask=(np.asarray(drivable_mask)>0).astype(np.uint8); l,t,w,h=content_rect
    if mask.ndim!=2 or w<=0 or h<=0:return None,mask
    content=mask[t:t+h,l:l+w]
    if content.shape!=(h,w) or not np.any(content):return None,mask
    k=min(7,max(3,int(round(w*.008))|1))
    clean_c=cv2.morphologyEx(content,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    cleaned=np.zeros_like(mask); cleaned[t:t+h,l:l+w]=clean_c
    seed=np.zeros_like(content,dtype=bool); seed[int(h*.70):int(h*.97),int(w*.39):int(w*.61)]=True
    candidates=np.argwhere((clean_c>0)&seed)
    if not len(candidates):return None,cleaned
    sy,sx=candidates[np.argmin(((candidates-np.array((h*.85,w*.5)))**2).sum(1))]
    filled=clean_c.copy(); cv2.floodFill(filled,np.zeros((h+2,w+2),np.uint8),(int(sx),int(sy)),2,flags=8)
    comp=filled==2
    if comp.mean()<.025 or np.count_nonzero(comp&seed)<max(6,int(seed.sum()*.008)):return None,cleaned
    has=comp.any(1); lo=np.argmax(comp,1); hi=w-1-np.argmax(comp[:,::-1],1); widths=hi-lo+1
    usable=has&(widths>=6); centers=(lo+hi)*.5; half=np.maximum(1,(widths*.38).astype(int)); xs=np.arange(w)[None,:]
    corridor=comp&usable[:,None]&(xs>=(np.rint(centers).astype(int)-half)[:,None])&(xs<=(np.rint(centers).astype(int)+half)[:,None])
    out=np.zeros_like(mask); out[t:t+h,l:l+w]=corridor; bounds=[None]*mask.shape[0]
    for y in np.flatnonzero(corridor.any(1)):
        row=np.flatnonzero(corridor[y]); bounds[t+int(y)]=(l+int(row[0]),l+int(row[-1]))
    coverage=float(corridor.any(1).mean())
    if coverage<.25:return None,cleaned
    quality=float(np.clip(.55*min(coverage/.65,1)+.45*min(comp.mean()/.30,1),0,1))
    out.setflags(write=False); cleaned.setflags(write=False)
    return CorridorGeometry(out,quality,tuple(bounds)),cleaned
