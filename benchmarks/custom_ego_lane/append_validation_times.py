#!/usr/bin/env python3
"""Append selected, deduplicated timestamps to an existing validation clip."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from build_dataset import dhash,hamming

def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--clip',required=True);p.add_argument('--timestamps',type=float,nargs='+',required=True);p.add_argument('--dedup-distance',type=int,default=10);a=p.parse_args();d=json.loads(a.dataset.read_text());items=[x for x in d['annotations'] if x['clip_id']==a.clip]
 if not items or any(x['split']!='validation' for x in items):raise SystemExit('existing validation clip required')
 source=next(s for s in d['sources'] if s['clip_id']==a.clip);video=Path(source['path']);cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width,height=d['video']['width'],d['video']['height'];hashes=[]
 for x in items:
  im=cv2.imread(str(a.dataset.parent/x['image']));hashes.append(dhash(im))
 next_id=max(x['item_id'] for x in d['annotations'])+1;added=[];rejected=[]
 for timestamp in a.timestamps:
  frame_id=min(count-1,max(0,round(timestamp*fps)));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,image=cap.read()
  if not ok:rejected.append({'timestamp':timestamp,'reason':'decode'});continue
  sig=dhash(image);distance=min(hamming(sig,old) for old in hashes)
  if distance<a.dedup_distance:rejected.append({'timestamp':timestamp,'reason':'near_duplicate','distance':distance});continue
  hashes.append(sig);relative=Path('frames')/f'validation_{a.clip}_{frame_id:06d}.jpg';cv2.imwrite(str(a.dataset.parent/relative),cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,94]);record={'item_id':next_id+len(added),'frame_id':frame_id,'timestamp':frame_id/fps,'image':str(relative),'source_video':str(video),'source_sha256':source['sha256'],'clip_id':a.clip,'split':'validation','selection_stratum':'unassigned','scene_type':[],'review_status':'pending','label_origin':'UNREVIEWED','ego_lane_status':None,'annotation_flag':None,'left_boundary':{'status':None,'points':[]},'right_boundary':{'status':None,'points':[]},'notes':'targeted diversity supplement'};d['annotations'].append(record);items.append(record);added.append(record)
 cap.release();times=sorted(x['timestamp'] for x in items);duration=count/fps;quarters=[sum(min(3,int(4*t/duration))==q for t in times) for q in range(4)];audit=source['sampling_audit'];audit.update(first_sample_s=round(times[0],3),last_sample_s=round(times[-1],3),samples=len(times),median_gap_s=round(float(np.median(np.diff(times))),3),quarter_counts=quarters);audit['near_duplicate_rejections']=audit.get('near_duplicate_rejections',0)+sum(x['reason']=='near_duplicate' for x in rejected);audit['supplemental_timestamps_s']=[round(x['timestamp'],3) for x in added];source['frames']=len(times);a.dataset.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'added':[{'item_id':x['item_id'],'timestamp':round(x['timestamp'],3)} for x in added],'rejected':rejected,'audit':audit},indent=2))
if __name__=='__main__':main()
