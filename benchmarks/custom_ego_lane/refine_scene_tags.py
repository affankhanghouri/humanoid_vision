#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def tags(x):
 clip=x['clip_id'];t=x['timestamp'];l=x['left_boundary']['status']=='visible';r=x['right_boundary']['status']=='visible'
 if clip=='urban_driving':
  if t<8:return ['clear_normal_lane','parked_cars_curb']
  if t<12:return ['intersection','crosswalk_stop_line','parked_cars_curb']
  if t<18:return ['clear_normal_lane','parked_cars_curb']
  if t<26:return ['intersection','crosswalk_stop_line','parked_cars_curb']
  if t<34:return ['intersection','parked_cars_curb','occlusion']
  return ['intersection','occlusion']
 if clip=='winter_residential':
  if t<35:return ['intersection','unmarked_road','parked_cars_curb']
  if t<90:return ['unmarked_road','parked_cars_curb','turning_curved_road']+(['occlusion'] if 70<=t<86 else [])
  return ['unmarked_road','intersection','crosswalk_stop_line']
 if clip=='urban_validation':
  result=['unmarked_road','parked_cars_curb']
  if t<18 or 42<=t<58 or 70<=t<91:result.append('intersection')
  if 18<=t<36 or 58<=t<78:result.append('turning_curved_road')
  if 8<=t<30 or 61<=t<75:result.append('occlusion')
  return result
 result=['clear_normal_lane','turning_curved_road']
 if l!=r:result.append('occlusion')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--clip');a=p.parse_args();d=json.loads(a.dataset.read_text())
 for x in d['annotations']:
  if a.clip and x['clip_id']!=a.clip:continue
  x['scene_type']=tags(x);x['selection_stratum']=x['scene_type'][0]
 a.dataset.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
