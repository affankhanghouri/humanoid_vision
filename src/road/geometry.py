"""Sparse road extents; these are not lane boundaries."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class RoadExtentSample:
    y: float
    left: float
    right: float
    center: float
    width: float
    confidence: float

@dataclass(frozen=True)
class RoadGeometry:
    source_frame_id: int
    source_timestamp: float
    transform_timestamp: float
    samples: tuple[RoadExtentSample,...]
    confidence: float
    projected: bool=False

def derive_road_geometry(mask,content_rect,source_frame_id,source_timestamp,row_count=16):
    binary=(np.asarray(mask)>0).astype(np.uint8);l,t,w,h=content_rect
    if binary.ndim!=2 or w<=0 or h<=0:return None
    content=binary[t:t+h,l:l+w]
    if content.shape!=(h,w):return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(content,8)
    if count<=1:return None
    seed=np.zeros((h,w),bool);seed[int(.72*h):int(.97*h),int(.42*w):int(.58*w)]=True
    best=max(range(1,count),key=lambda k:int(np.count_nonzero((labels==k)&seed)))
    component=labels==best;seed_overlap=np.count_nonzero(component&seed)/max(1,np.count_nonzero(seed))
    if seed_overlap<.015 or stats[best,cv2.CC_STAT_AREA]<.025*w*h:return None
    rows=np.rint(np.linspace(.40*h,.92*h,row_count)).astype(int);samples=[]
    for y in rows:
        xs=np.flatnonzero(component[min(max(y,0),h-1)])
        if len(xs)<.15*w:continue
        gaps=np.flatnonzero(np.diff(xs)>1);runs=np.split(xs,gaps+1)
        run=max(runs,key=len)
        if len(run)<.15*w:continue
        left,right=float(l+run[0]),float(l+run[-1]);width=right-left+1
        edge_conf=min(1.,width/(.55*w));samples.append(RoadExtentSample(float(t+y),left,right,(left+right)/2,width,edge_conf))
    if len(samples)<max(8,row_count//2):return None
    coverage=len(samples)/row_count;confidence=float(np.clip(.55*coverage+.25*min(seed_overlap/.35,1)+.20*min(stats[best,cv2.CC_STAT_AREA]/(.30*w*h),1),0,1))
    return RoadGeometry(source_frame_id,source_timestamp,source_timestamp,tuple(samples),confidence)

def conservative_corridor(geometry,width_fraction=.35):
    """Return sparse `(y,left,right)` corridor samples inside the road extents."""
    if geometry is None:return ()
    return tuple((s.y,s.center-s.width*width_fraction/2,s.center+s.width*width_fraction/2) for s in geometry.samples)

def project_road_geometry(geometry,motion,target_timestamp,max_age=1.05,max_extrapolation=.20):
    """Carry sparse samples with the existing global affine chain; fail closed."""
    if geometry is None or motion is None:return None
    age=target_timestamp-geometry.source_timestamp
    if age<0 or age>max_age:return None
    at=geometry.source_timestamp;matrix=np.eye(3);chain_target=min(target_timestamp,motion.timestamp)
    for step in motion.steps:
        if step.end<=at+1e-6:continue
        if step.start>at+1e-5 or step.end>chain_target+1e-5 or step.affine is None:return None
        part=np.eye(3);part[:2]=step.affine;matrix=part@matrix;at=step.end
        if at>=chain_target-1e-6:break
    if abs(at-chain_target)>1e-5:return None
    extra=target_timestamp-motion.timestamp
    if extra>1e-6:
        if extra>max_extrapolation or not motion.steps or motion.steps[-1].affine is None:return None
        step=motion.steps[-1];part=np.eye(3);part[:2]=np.eye(2,3)+extra/(step.end-step.start)*(step.affine-np.eye(2,3));matrix=part@matrix
    sx,sy=motion.scale;out=[]
    for s in geometry.samples:
        pts=np.array(((s.left*sx,s.y*sy,1),(s.right*sx,s.y*sy,1)))@matrix.T
        left,right=sorted((pts[0,0]/sx,pts[1,0]/sx));y=float(np.mean(pts[:,1]/sy));width=right-left
        out.append(RoadExtentSample(y,left,right,(left+right)/2,width,s.confidence))
    return RoadGeometry(geometry.source_frame_id,geometry.source_timestamp,target_timestamp,tuple(out),geometry.confidence,True)
