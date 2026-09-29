#!/usr/bin/env python3
"""Fill short highway proposal gaps only when bracketed by reviewed geometry candidates."""
import argparse,json
from pathlib import Path
import numpy as np
def sample(points,ys):
 a=np.asarray(points,float);o=np.argsort(a[:,1]);a=a[o];return np.interp(ys,a[:,1],a[:,0])
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();d=json.loads(a.dataset.read_text());h=d['video']['height'];ys=np.linspace(.55*h,.92*h,8);filled=0
 for clip in ('highway_project','highway_challenge'):
  items=[x for x in d['annotations'] if x['clip_id']==clip];good=[i for i,x in enumerate(items) if x['ego_lane_status']=='valid' and all(x[k]['status']=='visible' for k in ('left_boundary','right_boundary'))]
  for i,item in enumerate(items):
   if item['ego_lane_status']!='ambiguous':continue
   before=max((j for j in good if j<i),default=None);after=min((j for j in good if j>i),default=None)
   if before is None or after is None:continue
   left,right=items[before],items[after];span=right['timestamp']-left['timestamp']
   if span<=0 or span>1.2:continue
   ratio=(item['timestamp']-left['timestamp'])/span
   for key in ('left_boundary','right_boundary'):
    x0=sample(left[key]['points'],ys);x1=sample(right[key]['points'],ys);xs=x0+(x1-x0)*ratio;item[key]={'status':'visible','points':[[round(float(x),2),round(float(y),2)] for x,y in zip(xs,ys)]}
   item.update(ego_lane_status='valid',annotation_flag=None,label_origin='AUTO_TEMPORAL_HIGH_CONFIDENCE',notes='short gap interpolated between two high-confidence same-clip highway proposals; requires visual review');filled+=1
 a.dataset.write_text(json.dumps(d,indent=2)+'\n');print({'filled':filled})
if __name__=='__main__':main()
