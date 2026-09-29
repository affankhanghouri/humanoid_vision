#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
COLORS={'left_boundary':(0,255,255),'right_boundary':(255,100,0)}
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-sheet',type=int,default=40);a=p.parse_args();d=json.loads(a.dataset.read_text());a.output.mkdir(parents=True,exist_ok=True)
 for clip in sorted({x['clip_id'] for x in d['annotations']}):
  items=[x for x in d['annotations'] if x['clip_id']==clip]
  for page,start in enumerate(range(0,len(items),a.per_sheet)):
   tiles=[]
   for item in items[start:start+a.per_sheet]:
    im=cv2.imread(str(a.dataset.parent/item['image']))
    for key,color in COLORS.items():
     pts=np.asarray(item[key]['points'],np.int32)
     if len(pts)>=2:cv2.polylines(im,[pts],False,color,5,cv2.LINE_AA)
    text=f"#{item['item_id']} {item['timestamp']:.1f}s {item['ego_lane_status']} {item['label_origin'].replace('AUTO_','')}"
    cv2.rectangle(im,(0,0),(im.shape[1],42),(0,0,0),-1);cv2.putText(im,text,(8,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA);tiles.append(cv2.resize(im,(320,180)))
   while len(tiles)%5:tiles.append(np.zeros((180,320,3),np.uint8))
   sheet=np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]);cv2.imwrite(str(a.output/f'{clip}_{page:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,78])
if __name__=='__main__':main()
