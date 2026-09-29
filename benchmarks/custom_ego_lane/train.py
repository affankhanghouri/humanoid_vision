#!/usr/bin/env python3
"""Train only on completed HUMAN_REVIEWED local labels."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from data import EgoLaneDataset
from losses import ego_lane_loss
from model import TinyEgoLaneNet,parameter_count

def validation(model,loader,device):
 model.eval();probs=[];states=[];errors=[[],[]]
 with torch.no_grad():
  for image,x,visible,state,*_ in loader:
   px,_,logits=model(image.to(device));probs.extend(torch.softmax(logits,1)[:,0].cpu().numpy());states.extend(state.numpy())
   for side in range(2):
    mask=visible[:,:,side].bool()&(state[:,None]==0)
    errors[side].extend(torch.abs(px.cpu()[:,:,side]-x[:,:,side])[mask].tolist())
 probs=np.asarray(probs);states=np.asarray(states);actual_valid=states==0;actual_invalid=states==1;nv=int(actual_valid.sum());ni=int(actual_invalid.sum());candidates=np.unique(np.r_[.05,np.linspace(.1,.95,86),probs,.99]);best=None
 for threshold in candidates:
  pred=probs>=threshold;false=int(np.sum(pred&actual_invalid));missed=int(np.sum(~pred&actual_valid));score=3*false/max(ni,1)+missed/max(nv,1)
  row={'valid_threshold':float(threshold),'valid_lane_detection_rate':(nv-missed)/nv if nv else None,'false_lane_rate':false/ni if ni else None,'missed_lane_rate':missed/nv if nv else None,'correct_abstain_rate':(ni-false)/ni if ni else None,'classification_safety_score':score}
  if best is None or (score,row['false_lane_rate'],-row['valid_lane_detection_rate'])<(best['classification_safety_score'],best['false_lane_rate'],-best['valid_lane_detection_rate']):best=row
 best.update(records=len(states),left_boundary_error_width_fraction_mean=float(np.mean(errors[0])) if errors[0] else None,right_boundary_error_width_fraction_mean=float(np.mean(errors[1])) if errors[1] else None)
 return best

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--patience',type=int,default=8);p.add_argument('--freeze-backbone',action='store_true');p.add_argument('--seed',type=int,default=7);p.add_argument('--resume',type=Path);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--output',type=Path,default=Path(__file__).with_name('tiny_ego_lane.pt'));a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 train=EgoLaneDataset(a.manifest,'train',True);val=EgoLaneDataset(a.manifest,'validation',False);train_states={name:sum(x['ego_lane_status']==name for x in train.records) for name in ('valid','invalid','ambiguous')};val_states={name:sum(x['ego_lane_status']==name for x in val.records) for name in ('valid','invalid','ambiguous')}
 if len(train)<300 or len(val)<80 or val_states['valid']<20 or val_states['invalid']<20:raise SystemExit(f'REFUSED: need >=300 reviewed train, >=80 reviewed validation, and >=20 valid/invalid validation records; found train={len(train)} {train_states}, validation={len(val)} {val_states}')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=TinyEgoLaneNet(pretrained=not bool(a.resume)).to(device)
 if a.resume:model.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=True)['state_dict'])
 if a.freeze_backbone:
  for parameter in model.features.parameters():parameter.requires_grad=False
 opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr,weight_decay=1e-4);tl=DataLoader(train,a.batch_size,shuffle=True,num_workers=2);vl=DataLoader(val,a.batch_size,num_workers=2);history=[];best=float('inf');stale=0
 for epoch in range(a.epochs):
  model.train();running=[]
  if a.freeze_backbone:model.features.eval()
  for image,x,vis,state,weight,geometry,_ in tl:
   batch=[z.to(device) for z in (image,x,vis,state,weight,geometry)];loss=ego_lane_loss(model(batch[0]),*batch[1:4],sample_weight=batch[4],geometry_weight=batch[5]);opt.zero_grad();loss['total'].backward();opt.step();running.append(float(loss['total'].detach()))
  m=validation(model,vl,device);score=m['classification_safety_score']+(m['left_boundary_error_width_fraction_mean'] or 0)+(m['right_boundary_error_width_fraction_mean'] or 0);row={'epoch':epoch+1,'train_loss':float(np.mean(running)),'safety_score':score,**m};history.append(row);print(row,flush=True)
  if score<best-1e-6:
   best=score;stale=0;torch.save({'state_dict':model.state_dict(),'architecture':'TinyEgoLaneNet','parameters':parameter_count(model),'valid_threshold':m['valid_threshold'],'validation_metrics':m,'epoch':epoch+1},a.output)
  else:stale+=1
  a.output.with_suffix('.history.json').write_text(json.dumps({'stopped_early':False,'epochs':history},indent=2)+'\n')
  if stale>=a.patience:
   a.output.with_suffix('.history.json').write_text(json.dumps({'stopped_early':True,'patience':a.patience,'best_epoch':min(history,key=lambda x:x['safety_score'])['epoch'],'epochs':history},indent=2)+'\n');break
if __name__=='__main__':main()
