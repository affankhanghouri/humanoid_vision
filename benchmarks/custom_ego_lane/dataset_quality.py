#!/usr/bin/env python3
import argparse,collections,json
from pathlib import Path

def summarize(items):
 reviewed=[x for x in items if x.get('review_status')=='complete' and x.get('label_origin')=='HUMAN_REVIEWED'];states=collections.Counter(x.get('ego_lane_status') for x in reviewed);bounds=collections.Counter();prov=collections.Counter(x.get('label_origin') for x in items);scenes=collections.Counter(s for x in items for s in x.get('scene_type',[]));reviewed_scenes=collections.Counter(s for x in reviewed for s in x.get('scene_type',[]))
 for x in reviewed:
  l=x['left_boundary']['status']=='visible';r=x['right_boundary']['status']=='visible';bounds['both_visible' if l and r else 'left_visible_only' if l else 'right_visible_only' if r else 'neither_visible']+=1
 return {'total':len(items),'reviewed_total':len(reviewed),'valid':states['valid'],'invalid':states['invalid'],'ambiguous':states['ambiguous'],'left_visible_only':bounds['left_visible_only'],'right_visible_only':bounds['right_visible_only'],'both_visible':bounds['both_visible'],'neither_visible':bounds['neither_visible'],'human_reviewed':prov['HUMAN_REVIEWED'],'high_confidence_automatic':prov['AUTO_HIGH_CONFIDENCE'],'low_confidence_unreviewed':prov['AUTO_LOW_CONFIDENCE']+prov['UNREVIEWED'],'temporal_reviewed':sum(x.get('proposal_origin')=='TEMPORAL_INTERPOLATED' and x.get('label_origin')=='HUMAN_REVIEWED' for x in items),'rejected':prov['REJECTED'],'scene_types_all':dict(scenes),'scene_types_reviewed':dict(reviewed_scenes)}
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();d=json.loads(a.dataset.read_text());report={'train':summarize([x for x in d['annotations'] if x['split']=='train']),'validation':summarize([x for x in d['annotations'] if x['split']=='validation']),'by_source':{}}
 for clip in sorted({x['clip_id'] for x in d['annotations']}):report['by_source'][clip]=summarize([x for x in d['annotations'] if x['clip_id']==clip])
 text=json.dumps(report,indent=2)+'\n';print(text,end='')
 if a.output:a.output.write_text(text)
if __name__=='__main__':main()
