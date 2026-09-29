#!/usr/bin/env python3
"""Assign scene tags from the completed chronological target-camera audit."""
import argparse,json
from pathlib import Path

def tags(item):
 t=float(item['timestamp']); clip=item['clip_id']; out=[]
 if clip=='target_train':
  if 252<=t<396: out=['unmarked_road','turning_curved_road']
  elif 92<=t<119 or 584<=t<611 or 684<=t<704: out=['intersection','crosswalk_stop_line','occlusion']
  else:
   out=['clear_normal_lane']
   if 63<=t<84 or 144<=t<170 or 396<=t<584: out.append('turning_curved_road')
   if 160<=t<205 or t>=584: out.append('occlusion')
   if t>=584: out.append('parked_cars_curb')
   if 611<=t<684 or 704<=t: out.append('faded_lane')
 else:
  if t<9.5 or 43.5<=t<48 or 60<=t<66: out=['intersection','crosswalk_stop_line','occlusion']
  elif 28<=t<42: out=['occlusion','parked_cars_curb','faded_lane']
  else:
   out=['clear_normal_lane','parked_cars_curb']
   if 48<=t<60 or t>=66: out.append('turning_curved_road')
   if 18<=t<42: out.append('occlusion')
 return list(dict.fromkeys(out))

def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();d=json.loads(a.dataset.read_text())
 for item in d['annotations']: item['scene_type']=tags(item)
 d['scene_tag_method']='manual chronological windows from full train/validation contact-sheet audit'
 a.dataset.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
