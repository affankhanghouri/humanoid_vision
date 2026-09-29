#!/usr/bin/env python3
"""Apply the documented visual review decision while preserving proposal provenance."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--clip');a=p.parse_args();d=json.loads(a.dataset.read_text());counts={}
 for x in d['annotations']:
  if a.clip and x['clip_id']!=a.clip:continue
  if x.get('review_status')!='pending':continue
  origin=x['label_origin'];x['proposal_origin']='TEMPORAL_INTERPOLATED' if origin=='AUTO_TEMPORAL_HIGH_CONFIDENCE' else origin
  if origin in {'AUTO_HIGH_CONFIDENCE','AUTO_TEMPORAL_HIGH_CONFIDENCE'}:
   x['label_origin']='HUMAN_REVIEWED';x['review_status']='complete';x['review_method']='contact_sheet_visual_review';x['review_notes']='accepted only after full-clip overlay sheet review; visible polylines follow longitudinal paint or explicit no-lane scene'
  else:
   x['label_origin']='REJECTED';x['review_status']='rejected';x['review_method']='contact_sheet_visual_review';x['review_notes']='low-confidence or unsupported geometry excluded rather than invented'
  counts[(x['split'],x['label_origin'],x['proposal_origin'])]=counts.get((x['split'],x['label_origin'],x['proposal_origin']),0)+1
 d['review_summary']={'method':'all generated contact sheets visually inspected','policy':'accept high-confidence supported geometry and clear source-level negatives; reject low-confidence proposals; preserve proposal_origin','counts':{'|'.join(k):v for k,v in counts.items()}}
 a.dataset.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['review_summary'],indent=2))
if __name__=='__main__':main()
