#!/usr/bin/env python3
"""Conservative color/geometry proposals; never marks them human-reviewed."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
SCENES={
 'urban_driving':['parked_cars_curb','intersection','crosswalk_stop_line','occlusion'],
 'highway_project':['clear_normal_lane','turning_curved_road','occlusion'],
 'winter_residential':['unmarked_road','parked_cars_curb','occlusion','turning_curved_road'],
 'highway_challenge':['clear_normal_lane','turning_curved_road','occlusion'],
 'urban_validation':['unmarked_road','parked_cars_curb','intersection','occlusion'],
}
def fit_boundary(image,side):
 h,w=image.shape[:2];hls=cv2.cvtColor(image,cv2.COLOR_BGR2HLS);H,L,S=cv2.split(hls);white=(L>160)&(S<150)
 hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);yellow=cv2.inRange(hsv,np.array((10,70,70)),np.array((40,255,255)))>0
 mask=(yellow if side=='left' else white).astype(np.uint8)*255;roi=np.zeros_like(mask);cv2.fillPoly(roi,[np.array([(int(.03*w),h-1),(int(.39*w),int(.48*h)),(int(.61*w),int(.48*h)),(int(.97*w),h-1)])],255);mask&=roi;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));lines=cv2.HoughLinesP(cv2.Canny(mask,50,150),1,np.pi/180,20,minLineLength=18,maxLineGap=35);points=[]
 if lines is None:return []
 for x1,y1,x2,y2 in lines.reshape(-1,4):
  if y2==y1:continue
  slope=(x2-x1)/(y2-y1);mx=(x1+x2)/2
  if side=='left' and slope<-.10 and mx<w*.62:points.extend(((x1,y1),(x2,y2)))
  if side=='right' and slope>.10 and mx>w*.38:points.extend(((x1,y1),(x2,y2)))
 if len(points)<6:return []
 a=np.asarray(points,float);coef=np.polyfit(a[:,1],a[:,0],1);low=max(.52*h,a[:,1].min());high=min(.96*h,a[:,1].max())
 if high-low<.20*h:return []
 ys=np.linspace(low,high,8);xs=np.polyval(coef,ys);bottom=float(np.polyval(coef,.92*h))
 if np.any(xs<0)|np.any(xs>=w):return []
 if side=='left' and not .03*w<bottom<.52*w:return []
 if side=='right' and not .48*w<bottom<.97*w:return []
 return [[round(float(x),2),round(float(y),2)] for x,y in zip(xs,ys)]
def empty(status='not_visible'):return {'status':status,'points':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--clip');a=p.parse_args();d=json.loads(a.dataset.read_text());counts={}
 for item in d['annotations']:
  if a.clip and item['clip_id']!=a.clip:continue
  if item.get('review_status')!='pending' or item.get('label_origin')!='UNREVIEWED':continue
  image=cv2.imread(str(a.dataset.parent/item['image']));clip=item['clip_id'];t=item['timestamp'];item['scene_type']=SCENES[clip]
  if clip in {'winter_residential','urban_validation'} or (clip=='urban_driving' and t>=34):
   item.update(ego_lane_status='invalid',annotation_flag='NO_VALID_EGO_LANE',left_boundary=empty(),right_boundary=empty(),label_origin='AUTO_HIGH_CONFIDENCE',review_status='pending',notes='source-level negative proposal; requires visual review')
  elif clip=='urban_driving' and t>=30:
   item.update(ego_lane_status='ambiguous',annotation_flag=None,left_boundary=empty('ambiguous'),right_boundary=empty('ambiguous'),label_origin='AUTO_LOW_CONFIDENCE',review_status='pending',notes='intersection transition excluded pending review')
  else:
   left=fit_boundary(image,'left');right=fit_boundary(image,'right');lb={'status':'visible','points':left} if left else empty();rb={'status':'visible','points':right} if right else empty()
   if left and right:
    widths=[]
    for l,r in zip(left,right):widths.append(r[0]-l[0])
    plausible=min(widths)>20 and .18*image.shape[1]<(right[-1][0]-left[-1][0])<.75*image.shape[1]
   else:plausible=bool(left or right)
   status='valid' if plausible else 'ambiguous';origin='AUTO_HIGH_CONFIDENCE' if plausible else 'AUTO_LOW_CONFIDENCE'
   item.update(ego_lane_status=status,annotation_flag=None,left_boundary=lb if plausible else empty('ambiguous'),right_boundary=rb if plausible else empty('ambiguous'),label_origin=origin,review_status='pending',notes='color/Hough structured proposal; requires visual review')
  counts[(clip,item['ego_lane_status'],item['label_origin'])]=counts.get((clip,item['ego_lane_status'],item['label_origin']),0)+1
 a.dataset.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'counts':{'|'.join(k):v for k,v in counts.items()}},indent=2))
if __name__=='__main__':main()
