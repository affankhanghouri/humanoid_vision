#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
import cv2
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();d=json.loads(a.dataset.read_text());report=[]
 for source in d['sources']:
  items=[x for x in d['annotations'] if x['clip_id']==source['clip_id']];cap=cv2.VideoCapture(source['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release();ids=[x['frame_id'] for x in items];diffs=[b-a for a,b in zip(ids,ids[1:])];stride=min(diffs) if diffs else 1;candidates=(n-1)//stride+1;duration=n/fps;quarters=[0,0,0,0]
  for x in items:quarters[min(3,int(4*x['timestamp']/duration))]+=1
  row={'clip_id':source['clip_id'],'filename':Path(source['path']).name,'first_sample_s':round(items[0]['timestamp'],3),'last_sample_s':round(items[-1]['timestamp'],3),'duration_s':round(duration,3),'samples':len(items),'median_gap_s':round(statistics.median(diffs)/fps,3),'quarter_counts':quarters,'near_duplicate_rejections':candidates-len(items)};report.append(row);source['sampling_audit']=row
 d['sampling_audit']=report;a.dataset.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
