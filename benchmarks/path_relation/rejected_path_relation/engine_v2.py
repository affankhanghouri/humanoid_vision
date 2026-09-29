"""Geometry-only, motion-aligned path relation v2."""
from collections import deque
from dataclasses import dataclass,field
import math,time
import numpy as np
from rejected_path_relation.geometry import image_to_mask,integral_mask,integral_patch_fraction,corridor_distance_and_side
from rejected_path_relation.geometry_v2 import Probe,object_probes,compose_inverse,transform_point,construct_forward_corridor_v2
from rejected_path_relation.types import PathRelationState,RoadSupportKind,PathRelationObservationV2
from rejected_path_relation.engine import SUPPORTED_CLASSES

@dataclass
class _Hist:
    samples:deque=field(default_factory=lambda:deque(maxlen=10)); stable:PathRelationState=PathRelationState.UNKNOWN
    pending:PathRelationState|None=None; count:int=0; since:float=0.; last_seen:float=0.

class PathRelationEngineV2:
    def __init__(self,road_max_age=1.05,tracking_max_age=.75,track_expiry=2.,persistence=2):
        self.road_max_age=road_max_age; self.tracking_max_age=tracking_max_age; self.track_expiry=track_expiry
        self.persistence=persistence; self._tracks={}; self._key=None; self._corridor=None; self.cleaned_mask=None
        self._road_i=None; self._corr_i=None
    def _prepare(self,road):
        key=None if road is None else (road.source_frame_id,id(road.drivable_mask))
        if key!=self._key:
            self._key=key; self._corridor,self.cleaned_mask=(None,None) if road is None else construct_forward_corridor_v2(road.drivable_mask,road.content_rect)
            self._road_i=None if self.cleaned_mask is None else integral_mask(self.cleaned_mask)
            self._corr_i=None if self._corridor is None else integral_mask(self._corridor.mask)
        return self._corridor
    def _fraction(self,integral,probe,inverse,scale,frame_size,road):
        if not probe.usable:return None
        native=image_to_mask(transform_point(probe.center,inverse,scale),frame_size,road.content_rect,road.drivable_mask.shape)
        if native is None:return None
        _,_,cw,ch=road.content_rect; fw,fh=frame_size
        radii=(max(1,round(probe.radii[0]*cw/fw)),max(1,round(probe.radii[1]*ch/fh)))
        return integral_patch_fraction(integral,native,radii,road.drivable_mask.shape)
    @staticmethod
    def _support(vals):
        usable=[v for v in vals if v is not None]
        if len(usable)<2:return None,RoadSupportKind.UNKNOWN
        c,b,l,r=vals; surround=[v for v in (b,l,r) if v is not None]; weights=(.25,.35,.20,.20)
        score=sum(w*v for w,v in zip(weights,vals) if v is not None)/sum(w for w,v in zip(weights,vals) if v is not None)
        inferred=(b is not None and b>=.60 and sum(v>=.50 for v in (l,r) if v is not None)>=1) or sum(v>=.62 for v in (l,r) if v is not None)>=2
        if c is not None and c>=.52 and score>=.48:return score,RoadSupportKind.ROAD_VISIBLE_AND_SUPPORTED
        if inferred or (max(surround) >= .65 and score >= .12):return min(score,.85),RoadSupportKind.ROAD_INFERRED_FROM_SURROUNDINGS
        if len(usable)>=3 and score<=.30 and max(usable)<=.48:return score,RoadSupportKind.ROAD_NOT_SUPPORTED
        return score,RoadSupportKind.UNKNOWN
    def _stabilize(self,h,candidate,new,timestamp):
        if not new:return h.stable
        required=1 if h.stable==PathRelationState.UNKNOWN and candidate in (PathRelationState.OFF_DRIVABLE,PathRelationState.ON_DRIVABLE_OUTSIDE_CORRIDOR,PathRelationState.IN_CORRIDOR) else self.persistence
        if candidate==h.stable:h.pending=None;h.count=0;return h.stable
        if h.pending==candidate:h.count+=1
        else:h.pending=candidate;h.count=1
        if h.count>=required:h.stable=candidate;h.pending=None;h.count=0;h.since=timestamp
        return h.stable
    def _unknown(self,e,state,road,now,contact,corridor):
        return PathRelationObservationV2(e.entity_id,state.source_frame_id,state.source_timestamp,
            None if road is None else road.source_frame_id,None if road is None else road.source_timestamp,
            None,now,contact,None,(),None,RoadSupportKind.UNKNOWN,None,PathRelationState.UNKNOWN,0.,False,
            False,False,0. if corridor is None else corridor.quality,0)
    def update(self,state,road,*,frame_id,timestamp,frame_size,motion=None):
        now=time.perf_counter(); active=set() if state is None else {e.entity_id for e in state.entities}
        for k in list(self._tracks):
            if k not in active and timestamp-self._tracks[k].last_seen>self.track_expiry:del self._tracks[k]
        if state is None:return ()
        track_ok=state.source_frame_id<=frame_id and 0<=timestamp-state.source_timestamp<=self.tracking_max_age
        road_age=math.inf if road is None else timestamp-road.source_timestamp
        fresh=road is not None and 0<=road_age<=self.road_max_age and road.source_frame_id<=frame_id
        corridor=self._prepare(road) if fresh else None; inverse=None
        if fresh:
            inverse=np.eye(3) if abs(timestamp-road.source_timestamp)<1e-6 else compose_inverse(motion,road.source_timestamp,timestamp,self.road_max_age)
        aligned=fresh and corridor is not None and inverse is not None
        out=[]
        for e in state.entities:
            h=self._tracks.setdefault(e.entity_id,_Hist()); h.last_seen=timestamp
            age=max(0.,timestamp-state.source_timestamp) if track_ok else 0.; cx=e.center_x+e.velocity_x*age; cy=e.center_y+e.velocity_y*age
            box=(cx-e.width/2,cy-e.height/2,cx+e.width/2,cy+e.height/2); contact=((box[0]+box[2])*.5,float(np.clip(box[3],0,frame_size[1]-1)))
            new=not h.samples or h.samples[-1][0]!=state.source_frame_id
            if not(e.class_name.lower() in SUPPORTED_CLASSES and track_ok and aligned):out.append(self._unknown(e,state,road,now,contact,corridor));continue
            scale=(1.,1.) if motion is None else motion.scale; probes=object_probes(box,frame_size)
            vals=tuple(self._fraction(self._road_i,p,inverse,scale,frame_size,road) for p in probes); score,kind=self._support(vals)
            bw,bh=box[2]-box[0],box[3]-box[1]
            footprint=(Probe("foot",((box[0]+box[2])*.5,box[3]-.02*bh),(.38*bw,.04*bh),0<box[3]<frame_size[1]),)
            ovs=[self._fraction(self._corr_i,p,inverse,scale,frame_size,road) for p in footprint]; valid=[v for v in ovs if v is not None]
            overlap=float(np.mean(valid)) if valid else None; prev=h.samples[-1] if h.samples else None
            road_on=kind in (RoadSupportKind.ROAD_VISIBLE_AND_SUPPORTED,RoadSupportKind.ROAD_INFERRED_FROM_SURROUNDINGS)
            was_in=h.stable in (PathRelationState.IN_CORRIDOR,PathRelationState.CROSSING_CORRIDOR,PathRelationState.LEAVING_CORRIDOR)
            inside=overlap is not None and overlap>=(.24 if was_in else .42)
            native=image_to_mask(transform_point(contact,inverse,scale),frame_size,road.content_rect,road.drivable_mask.shape)
            relation=corridor_distance_and_side(native,corridor,frame_size[0],road.content_rect[2]) if native else None
            motion_ok=e.misses==0 and e.last_observed_timestamp is not None and timestamp-e.last_observed_timestamp<=.45 and len(h.samples)>=2
            candidate=PathRelationState.UNKNOWN
            if kind==RoadSupportKind.ROAD_NOT_SUPPORTED:candidate=PathRelationState.OFF_DRIVABLE
            elif road_on:
                candidate=PathRelationState.IN_CORRIDOR if inside else PathRelationState.ON_DRIVABLE_OUTSIDE_CORRIDOR
                if motion_ok and prev is not None and relation is not None:
                    dist,side=relation; pd,ps=prev[3],prev[4]; lateral=abs(e.velocity_x)>=frame_size[0]*.025
                    if lateral and inside and ps in (-1,1) and side in (0,-ps):candidate=PathRelationState.CROSSING_CORRIDOR
                    elif not inside and pd is not None and dist<pd-frame_size[0]*.01:candidate=PathRelationState.ENTERING_CORRIDOR
                    elif was_in and not inside:candidate=PathRelationState.LEAVING_CORRIDOR
            if new:h.samples.append((state.source_frame_id,score,overlap,None if relation is None else relation[0],None if relation is None else relation[1]))
            final=self._stabilize(h,candidate,new,timestamp)
            conf=0. if final==PathRelationState.UNKNOWN or score is None or overlap is None else float(np.clip(.45*abs(score-.4)/.6+.30*abs(overlap-.33)/.67+.15*corridor.quality+.10*(.85 if kind==RoadSupportKind.ROAD_INFERRED_FROM_SURROUNDINGS else 1),0,1))
            pred=(contact[0]+e.velocity_x*.4,contact[1]+e.velocity_y*.4) if motion_ok else None
            out.append(PathRelationObservationV2(e.entity_id,state.source_frame_id,state.source_timestamp,road.source_frame_id,road.source_timestamp,timestamp,now,contact,pred,tuple((p.name,v) for p,v in zip(probes,vals)),score,kind,overlap,final,conf,motion_ok,True,abs(timestamp-road.source_timestamp)>1e-6,corridor.quality,len(h.samples)))
        return tuple(out)
    @property
    def corridor(self):return self._corridor
    @property
    def tracked_ids(self):return frozenset(self._tracks)
