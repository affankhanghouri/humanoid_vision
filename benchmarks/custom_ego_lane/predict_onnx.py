#!/usr/bin/env python3
"""Convert a fixed trained ONNX model into the common lane prediction format."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from data import ANCHOR_Y,MEAN,STD
STATE=['valid','invalid','ambiguous']
def prep(image):
 image=cv2.cvtColor(cv2.resize(image,(320,192)),cv2.COLOR_BGR2RGB);return np.ascontiguousarray(((image.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None])
def main():
 p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);p.add_argument('--visibility-threshold',type=float,default=.5);p.add_argument('--valid-threshold',type=float,required=True);p.add_argument('--model-name',default='custom_ego_lane_mobilenetv3_small');a=p.parse_args();d=json.loads(a.dataset.read_text());w,h=d['video']['width'],d['video']['height'];session=ort.InferenceSession(str(a.model),providers=['CPUExecutionProvider']);name=session.get_inputs()[0].name;predictions=[]
 for item in d['annotations']:
  image=cv2.imread(str(a.dataset.parent/item['image']));coords,vis_logits,state_logits=session.run(None,{name:prep(image)});vis=1/(1+np.exp(-vis_logits[0]));prob=np.exp(state_logits[0]-state_logits[0].max());prob/=prob.sum();state='valid' if float(prob[0])>=a.valid_threshold else 'invalid';boundaries=[]
  for side in range(2):
   points=[[float(coords[0,row,side]*w),float(ANCHOR_Y[row]*h)] for row in range(len(ANCHOR_Y)) if vis[row,side]>=a.visibility_threshold]
   boundaries.append({'status':'visible' if len(points)>=2 else 'not_visible','points':points if len(points)>=2 else [],'confidence':float(vis[:,side].max())})
  if state=='invalid':boundaries=[{'status':'not_visible','points':[],'confidence':float(1)} for _ in range(2)]
  elif state=='valid' and not any(x['status']=='visible' for x in boundaries):state='ambiguous'
  predictions.append({'frame_id':item['frame_id'],'ego_lane_status':state,'left_boundary':boundaries[0],'right_boundary':boundaries[1]})
 a.output.write_text(json.dumps({'model':{'name':a.model_name,'path':str(a.model)},'valid_threshold':a.valid_threshold,'visibility_threshold':a.visibility_threshold,'predictions':predictions},indent=2)+'\n')
if __name__=='__main__':main()
