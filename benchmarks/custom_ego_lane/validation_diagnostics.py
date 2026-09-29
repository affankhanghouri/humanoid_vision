#!/usr/bin/env python3
"""Detailed development-split metrics and visual failure sheets."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lane_acceptance'))
from common import _polyline_error,_error_summary
from data import ANCHOR_Y,EgoLaneDataset
from model import TinyEgoLaneNet

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--visibility-threshold',type=float,default=.5);p.add_argument('--split',default='validation',choices=('train','validation'));p.add_argument('--no-sheets',action='store_true');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);dataset=EgoLaneDataset(a.manifest,a.split,False);model=TinyEgoLaneNet(False);checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval();threshold=float(checkpoint['valid_threshold']);w,h=dataset.width,dataset.height;counts={'frames':0,'valid':0,'invalid':0,'detected_valid':0,'false_lane':0,'missed':0,'correct_abstain':0};boundary={s:{'truth_visible':0,'detected':0,'errors':[],'matched':0,'no_overlap':0,'false_visible':0,'truth_not_visible':0} for s in ('left','right')};rows=[]
 with torch.no_grad():
  for index,item in enumerate(dataset.records):
   image_tensor,*_=dataset[index];coords,vis_logits,state_logits=model(image_tensor[None]);valid_prob=float(torch.softmax(state_logits,1)[0,0]);pred_valid=valid_prob>=threshold;truth_valid=item['ego_lane_status']=='valid';counts['frames']+=1;counts['valid']+=truth_valid;counts['invalid']+=not truth_valid;counts['detected_valid']+=pred_valid and truth_valid;counts['false_lane']+=pred_valid and not truth_valid;counts['missed']+=(not pred_valid) and truth_valid;counts['correct_abstain']+=(not pred_valid) and not truth_valid;vis=torch.sigmoid(vis_logits)[0].numpy();coords=coords[0].numpy();pred_bounds={};frame_errors=[]
   for side_idx,side in enumerate(('left','right')):
    pts=[[float(coords[r,side_idx]*w),float(ANCHOR_Y[r]*h)] for r in range(len(ANCHOR_Y)) if pred_valid and vis[r,side_idx]>=a.visibility_threshold];pred_visible=len(pts)>=2;pred_bounds[side]={'status':'visible' if pred_visible else 'not_visible','points':pts if pred_visible else []};truth=item[f'{side}_boundary'];state=boundary[side]
    if truth['status']=='visible':
     state['truth_visible']+=1
     if pred_visible:
      state['detected']+=1;values,overlap=_polyline_error(truth['points'],pts,w);state['matched']+=1;state['no_overlap']+=not overlap;state['errors'].extend(values.tolist());frame_errors.extend(values.tolist())
    else:
     state['truth_not_visible']+=1;state['false_visible']+=pred_visible
   rows.append({'item_id':item['item_id'],'clip_id':item['clip_id'],'timestamp':item['timestamp'],'truth':item['ego_lane_status'],'predicted':'valid' if pred_valid else 'invalid','valid_probability':valid_prob,'mean_boundary_error_px':float(np.mean(frame_errors)) if frame_errors else None,'boundaries':pred_bounds})
 metrics={'valid_lane_detection_rate':counts['detected_valid']/counts['valid'],'false_lane_rate':counts['false_lane']/counts['invalid'],'missed_lane_rate':counts['missed']/counts['valid'],'correct_abstain_rate':counts['correct_abstain']/counts['invalid'],'valid_threshold':threshold,'visibility_threshold':a.visibility_threshold}
 for side in ('left','right'):
  s=boundary[side];metrics[f'{side}_boundary_detection_rate']=s['detected']/s['truth_visible'] if s['truth_visible'] else None;metrics[f'{side}_false_visibility_rate']=s['false_visible']/s['truth_not_visible'] if s['truth_not_visible'] else None;metrics[f'{side}_boundary_error']=_error_summary(s['errors'],w,s['matched'],s['no_overlap'])
 payload={'checkpoint':str(a.checkpoint),'counts':counts,'metrics':metrics,'predictions':rows};(a.output_dir/f'{a.split}_metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
 if a.no_sheets:
  print(json.dumps({'counts':counts,'metrics':metrics},indent=2));return
 ordered=sorted(rows,key=lambda x:(x['predicted']==x['truth'], -(x['mean_boundary_error_px'] or 0)));manifest=json.loads(a.manifest.read_text());by_id={x['item_id']:x for x in manifest['annotations']};tiles=[]
 for row in ordered:
  item=by_id[row['item_id']];im=cv2.imread(str(a.manifest.parent/item['image']))
  for side,color in (('left',(0,255,255)),('right',(255,100,0))):
   truth=np.asarray(item[f'{side}_boundary']['points'],np.int32)
   if len(truth)>=2:cv2.polylines(im,[truth],False,color,5,cv2.LINE_AA)
   pred=np.asarray(row['boundaries'][side]['points'],np.int32)
   if len(pred)>=2:cv2.polylines(im,[pred],False,(255,0,255) if side=='left' else (0,255,0),3,cv2.LINE_AA)
  label=f"#{row['item_id']} {row['truth']}->{row['predicted']} p={row['valid_probability']:.2f}";cv2.rectangle(im,(0,0),(im.shape[1],45),(0,0,0),-1);cv2.putText(im,label,(8,31),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA);tiles.append(cv2.resize(im,(320,180)))
 for page,start in enumerate(range(0,len(tiles),30)):
  page_tiles=tiles[start:start+30]
  while len(page_tiles)<30:page_tiles.append(np.zeros((180,320,3),np.uint8))
  sheet=np.vstack([np.hstack(page_tiles[i:i+5]) for i in range(0,30,5)]);cv2.imwrite(str(a.output_dir/f'validation_failures_{page:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,82])
 print(json.dumps({'counts':counts,'metrics':metrics},indent=2))
if __name__=='__main__':main()
