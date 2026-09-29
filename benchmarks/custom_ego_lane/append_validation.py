#!/usr/bin/env python3
"""Append one independent validation video without rebuilding existing splits."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
from build_dataset import FROZEN, dhash, hamming, sampling_stride

def fingerprint(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(1<<20): digest.update(block)
    return digest.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('video',type=Path);p.add_argument('--clip-id',required=True);p.add_argument('--limit',type=int,default=40);p.add_argument('--interval',type=float,default=.5);p.add_argument('--dedup-distance',type=int,default=10);a=p.parse_args()
    dataset=a.dataset.resolve();video=a.video.resolve();payload=json.loads(dataset.read_text())
    if a.clip_id in {s['clip_id'] for s in payload['sources']}: raise SystemExit(f'REFUSED: clip_id already exists: {a.clip_id}')
    frozen=json.loads(FROZEN.read_text());frozen_video=(FROZEN.parents[2]/frozen['video']['path']).resolve()
    if video==frozen_video: raise SystemExit('REFUSED: frozen acceptance video is quarantined')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise FileNotFoundError(video)
    fps=float(cap.get(cv2.CAP_PROP_FPS));frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));stride=sampling_stride(frame_count,fps,a.interval,a.limit)
    width,height=payload['video']['width'],payload['video']['height'];frames_dir=dataset.parent/'frames';frames_dir.mkdir(parents=True,exist_ok=True);source_hash=fingerprint(video)
    accepted=[];records=[];candidate_count=duplicate_rejections=0;next_id=max((x['item_id'] for x in payload['annotations']),default=-1)+1;frame_id=-1
    while len(records)<a.limit:
        ok,image=cap.read()
        if not ok: break
        frame_id+=1
        if frame_id%stride: continue
        candidate_count+=1;signature=dhash(image)
        if accepted and min(hamming(signature,old) for old in accepted[-20:])<a.dedup_distance:
            duplicate_rejections+=1;continue
        accepted.append(signature);resized=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA);relative=Path('frames')/f'validation_{a.clip_id}_{frame_id:06d}.jpg';cv2.imwrite(str(dataset.parent/relative),resized,[cv2.IMWRITE_JPEG_QUALITY,94])
        records.append({'item_id':next_id+len(records),'frame_id':frame_id,'timestamp':frame_id/fps,'image':str(relative),'source_video':str(video),'source_sha256':source_hash,'clip_id':a.clip_id,'split':'validation','selection_stratum':'unassigned','scene_type':[],'review_status':'pending','label_origin':'UNREVIEWED','ego_lane_status':None,'annotation_flag':None,'left_boundary':{'status':None,'points':[]},'right_boundary':{'status':None,'points':[]},'notes':''})
    cap.release()
    if not records: raise SystemExit('no frames were extracted')
    payload['annotations'].extend(records);duration=frame_count/fps;quarters=[0,0,0,0]
    for x in records: quarters[min(3,int(4*x['timestamp']/duration))]+=1
    audit={'clip_id':a.clip_id,'filename':video.name,'first_sample_s':round(records[0]['timestamp'],3),'last_sample_s':round(records[-1]['timestamp'],3),'duration_s':round(duration,3),'samples':len(records),'median_gap_s':round(float(np.median(np.diff([x['timestamp'] for x in records]))),3),'quarter_counts':quarters,'near_duplicate_rejections':duplicate_rejections,'candidate_slots_considered':candidate_count}
    payload['sources'].append({'path':str(video),'sha256':source_hash,'clip_id':a.clip_id,'split':'validation','frames':len(records),'sampling_audit':audit});dataset.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
