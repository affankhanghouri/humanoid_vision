#!/usr/bin/env python3
"""Export trained candidate, or explicit untrained graph for CPU feasibility only."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,onnx,onnxruntime as ort,torch
from data import MEAN,STD
from model import INPUT_SIZE,TinyEgoLaneNet,parameter_count

def prep(image):
 image=cv2.cvtColor(cv2.resize(image,(INPUT_SIZE[1],INPUT_SIZE[0])),cv2.COLOR_BGR2RGB);return np.ascontiguousarray(((image.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None])
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--allow-untrained-timing-only',action='store_true');p.add_argument('--output',type=Path,default=Path(__file__).with_name('tiny_ego_lane_untrained.onnx'));a=p.parse_args()
 if not a.checkpoint and not a.allow_untrained_timing_only:raise SystemExit('checkpoint required unless --allow-untrained-timing-only is explicit')
 torch.manual_seed(7);model=TinyEgoLaneNet(False).eval();trained=False;checkpoint={}
 if a.checkpoint:checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);trained=True
 sample=torch.randn(1,3,*INPUT_SIZE);torch.onnx.export(model,sample,a.output,input_names=['image'],output_names=['coordinates','visibility_logits','state_logits'],opset_version=17,dynamo=False);graph=onnx.load(a.output);onnx.checker.check_model(graph);session=ort.InferenceSession(str(a.output),providers=['CPUExecutionProvider']);checks=[]
 test_inputs=[('random',sample.numpy())]
 if a.manifest:
  d=json.loads(a.manifest.read_text());items=[x for x in d['annotations'] if x['split']=='validation' and x['review_status']=='complete' and x.get('label_origin')=='HUMAN_REVIEWED'][:10]
  test_inputs.extend((f"item_{x['item_id']}",prep(cv2.imread(str(a.manifest.parent/x['image'])))) for x in items)
 maxima={name:[] for name in ('coordinates','visibility_logits','state_logits')}
 for sample_name,array in test_inputs:
  actual=session.run(None,{'image':array});expected=[x.detach().numpy() for x in model(torch.from_numpy(array))]
  for name,left,right in zip(maxima,expected,actual):maxima[name].append(float(np.max(np.abs(left-right))))
 for name,values in maxima.items():checks.append({'output':name,'samples':len(values),'max_abs_error':max(values),'mean_max_abs_error':float(np.mean(values)),'allclose':max(values)<1e-4})
 result={'trained':trained,'checkpoint':str(a.checkpoint) if a.checkpoint else None,'parameters':parameter_count(model),'input':[1,3,*INPUT_SIZE],'onnx_checker':'passed','valid_threshold':checkpoint.get('valid_threshold'),'visibility_threshold':.5,'numerical':checks};a.output.with_suffix('.verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
