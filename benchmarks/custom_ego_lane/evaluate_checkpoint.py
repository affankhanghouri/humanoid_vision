#!/usr/bin/env python3
"""Evaluate a checkpoint and calibrate a safety-weighted valid-lane threshold."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from data import EgoLaneDataset
from model import TinyEgoLaneNet

def metrics(valid_probability,states,coord_errors,threshold):
 pred_valid=valid_probability>=threshold;actual_valid=states==0;actual_invalid=states==1
 false=int(np.sum(pred_valid&actual_invalid));missed=int(np.sum(~pred_valid&actual_valid));nv=int(actual_valid.sum());ni=int(actual_invalid.sum())
 return {'threshold':float(threshold),'records':int(len(states)),'valid_records':int(nv),'invalid_records':int(ni),'valid_lane_detection_rate':float((nv-missed)/nv) if nv else None,'false_lane_rate':float(false/ni) if ni else None,'missed_lane_rate':float(missed/nv) if nv else None,'correct_abstain_rate':float((ni-false)/ni) if ni else None,'left_boundary_error_width_fraction_mean':float(np.mean(coord_errors[0])) if coord_errors[0] else None,'right_boundary_error_width_fraction_mean':float(np.mean(coord_errors[1])) if coord_errors[1] else None,'safety_score':float(3*false/max(ni,1)+missed/max(nv,1))}
def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();dataset=EgoLaneDataset(a.manifest,'validation',False);loader=DataLoader(dataset,16,False,num_workers=2);model=TinyEgoLaneNet(False);model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['state_dict']);model.eval();probs=[];states=[];errors=[[],[]]
 with torch.no_grad():
  for image,x,visible,state,*_ in loader:
   px,_,logits=model(image);probs.extend(torch.softmax(logits,1)[:,0].numpy());states.extend(state.numpy())
   for side in range(2):
    mask=visible[:,:,side].bool()&(state[:,None]==0)
    errors[side].extend(torch.abs(px[:,:,side]-x[:,:,side])[mask].numpy().tolist())
 probs=np.asarray(probs);states=np.asarray(states);thresholds=np.unique(np.r_[np.linspace(.05,.95,91),probs,0.99]);sweep=[metrics(probs,states,errors,t) for t in thresholds];best=min(sweep,key=lambda x:(x['safety_score'],x['false_lane_rate'],-x['valid_lane_detection_rate']))
 result={'selection_rule':'minimize 3*false_lane_rate + missed_lane_rate; ties prefer fewer false lanes then higher recall','selected':best,'probability_summary':{'valid_ground_truth':{k:float(v) for k,v in zip(('min','median','p95','max'),(probs[states==0].min(),np.median(probs[states==0]),np.percentile(probs[states==0],95),probs[states==0].max()))},'invalid_ground_truth':{k:float(v) for k,v in zip(('min','median','p95','max'),(probs[states==1].min(),np.median(probs[states==1]),np.percentile(probs[states==1],95),probs[states==1].max()))}},'representative_thresholds':[metrics(probs,states,errors,t) for t in (.5,.6,.7,.8,.9,best['threshold'])]}
 text=json.dumps(result,indent=2)+'\n';print(text,end='');(a.output or a.checkpoint.with_suffix('.validation.json')).write_text(text)
if __name__=='__main__':main()
