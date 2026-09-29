#!/usr/bin/env python3
"""Recorded human visual-review decisions for the new urban validation clip."""
import argparse,json
from pathlib import Path
# Frames visually confirmed to have no trustworthy ego-lane boundary structure.
REVIEWED_INVALID_IDS={647,648,649,650,653,654,656,663,664,665,666,669,670,674,676,682,688,689,691,694}
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args();d=json.loads(a.dataset.read_text());counts={'reviewed_invalid':0,'rejected':0}
 for x in d['annotations']:
  if x['clip_id']!='urban_validation' or x['review_status']!='pending':continue
  x['proposal_origin']=x['label_origin']
  if x['item_id'] in REVIEWED_INVALID_IDS:
   x.update(ego_lane_status='invalid',annotation_flag='NO_VALID_EGO_LANE',left_boundary={'status':'not_visible','points':[]},right_boundary={'status':'not_visible','points':[]},label_origin='HUMAN_REVIEWED',review_status='complete',review_method='contact_sheet_visual_review',review_notes='full-resolution frame checked: no trustworthy painted ego-lane boundary; abstention is required')
   counts['reviewed_invalid']+=1
  else:
   x.update(label_origin='REJECTED',review_status='rejected',review_method='contact_sheet_visual_review',review_notes='automatic invalid proposal rejected: visible centerline or scene remains ambiguous without boundary annotation')
   counts['rejected']+=1
 a.dataset.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
