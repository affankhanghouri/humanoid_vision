"""Reviewed annotation loading and geometry-safe augmentation."""
import json, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from model import INPUT_SIZE, ROW_ANCHORS
ANCHOR_Y=np.linspace(.35,.95,ROW_ANCHORS,dtype=np.float32)
STATE_INDEX={"valid":0,"invalid":1,"ambiguous":2}
MEAN=np.asarray((.485,.456,.406),np.float32).reshape(1,1,3); STD=np.asarray((.229,.224,.225),np.float32).reshape(1,1,3)
def interpolate_boundary(boundary,width,height):
 x=np.zeros(ROW_ANCHORS,np.float32); visible=np.zeros(ROW_ANCHORS,np.float32)
 if boundary["status"]!="visible" or len(boundary["points"])<2:return x,visible
 points=np.asarray(boundary["points"],np.float32);points=points[np.argsort(points[:,1])];y=points[:,1]/height
 unique_y,inverse=np.unique(y,return_inverse=True);mean_x=np.asarray([points[inverse==i,0].mean()/width for i in range(len(unique_y))])
 mask=(ANCHOR_Y>=unique_y[0])&(ANCHOR_Y<=unique_y[-1]);x[mask]=np.interp(ANCHOR_Y[mask],unique_y,mean_x).clip(0,1);visible[mask]=1
 return x,visible
def flip_labels(coordinates, visible):
 return np.stack((1-coordinates[:,1],1-coordinates[:,0]),1), visible[:,::-1].copy()

class EgoLaneDataset(Dataset):
 def __init__(self,manifest:Path,split:str,augment:bool):
  self.manifest=manifest;payload=json.loads(manifest.read_text());self.width,self.height=payload["video"]["width"],payload["video"]["height"]
  self.records=[x for x in payload["annotations"] if x["split"]==split and x["review_status"]=="complete" and x.get("label_origin")=="HUMAN_REVIEWED"];self.augment=augment
 def __len__(self):return len(self.records)
 def __getitem__(self,index):
  item=self.records[index];image=cv2.imread(str(self.manifest.parent/item["image"]));
  if image is None:raise FileNotFoundError(item["image"])
  lx,lv=interpolate_boundary(item["left_boundary"],self.width,self.height);rx,rv=interpolate_boundary(item["right_boundary"],self.width,self.height)
  coordinates=np.stack((lx,rx),1);visible=np.stack((lv,rv),1)
  if self.augment and random.random()<.5:image=cv2.flip(image,1);coordinates,visible=flip_labels(coordinates,visible)
  if self.augment:
   image=cv2.convertScaleAbs(image,alpha=random.uniform(.85,1.15),beta=random.uniform(-12,12))
   if random.random()<.15:image=cv2.GaussianBlur(image,(3,3),0)
  image=cv2.cvtColor(cv2.resize(image,(INPUT_SIZE[1],INPUT_SIZE[0])),cv2.COLOR_BGR2RGB);image=(image.astype(np.float32)/255-MEAN)/STD
  state=STATE_INDEX[item["ego_lane_status"]];geometry=0. if state==2 else 1.;weight=.5 if state==2 else 1.
  return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),torch.from_numpy(coordinates),torch.from_numpy(visible),torch.tensor(state),torch.tensor(weight),torch.tensor(geometry),item["item_id"]
