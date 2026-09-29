#!/usr/bin/env python3
import argparse,collections,json
from pathlib import Path
from common import validate_annotation
def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();d=json.loads(a.manifest.read_text());w,h=d['video']['width'],d['video']['height'];errors=[];clips={}
 for source in d['sources']:
  if source['clip_id'] in clips:errors.append('duplicate clip_id '+source['clip_id'])
  clips[source['clip_id']]=source['split']
 if set(clips.values())!={'train','validation'}:errors.append('independent train and validation clips are required')
 counts=collections.Counter();review=collections.Counter()
 for item in d['annotations']:
  if clips.get(item['clip_id'])!=item['split']:errors.append(f"item {item['item_id']}: clip/split mismatch")
  review[item['review_status']]+=1
  if item['review_status']=='complete':
   errors.extend(validate_annotation(item,w,h,True));counts[item['ego_lane_status']]+=1
   lv=item['left_boundary']['status']=='visible';rv=item['right_boundary']['status']=='visible'
   if lv and not rv:counts['left_only']+=1
   if rv and not lv:counts['right_only']+=1
 print(json.dumps({'records':len(d['annotations']),'review':review,'labels':counts,'errors':errors},default=dict,indent=2))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
