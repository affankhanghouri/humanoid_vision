#!/usr/bin/env python3
"""CPU gate for the tiny graph. Untrained graph timings do not imply quality."""
import argparse,json,statistics,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];MEAN=np.array((.485,.456,.406),np.float32).reshape(1,1,3);STD=np.array((.229,.224,.225),np.float32).reshape(1,1,3)
def prep(im):
 im=cv2.cvtColor(cv2.resize(im,(320,192)),cv2.COLOR_BGR2RGB);return np.ascontiguousarray(((im.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None])
def stats(x):return {"mean_ms":statistics.fmean(x),"median_ms":statistics.median(x),"p95_ms":float(np.percentile(x,95)),"p99_ms":float(np.percentile(x,99))}
def ort_factory(path,threads):
 import onnxruntime as ort;o=ort.SessionOptions();o.intra_op_num_threads=threads;o.inter_op_num_threads=1;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;s=ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name;return lambda x:s.run(None,{name:x})
def ov_factory(path,threads):
 import openvino as ov;c=ov.Core().compile_model(str(path),'CPU',{'PERFORMANCE_HINT':'LATENCY','INFERENCE_NUM_THREADS':threads,'NUM_STREAMS':'1'});r=c.create_infer_request();i,o=c.input(0),list(c.outputs);return lambda x:(r.infer({i:x}),[np.array(r.get_tensor(z).data,copy=True) for z in o])[1]
def run(label,infer,images,warmup):
 for i in range(warmup):infer(prep(images[i%len(images)]))
 pre=[];inf=[];post=[];total=[]
 for im in images:
  a=time.perf_counter_ns();x=prep(im);b=time.perf_counter_ns();out=infer(x);c=time.perf_counter_ns();
  _=[1/(1+np.exp(-out[1])),np.argmax(out[2],axis=1)];d=time.perf_counter_ns();pre.append((b-a)/1e6);inf.append((c-b)/1e6);post.append((d-c)/1e6);total.append((d-a)/1e6)
 result={'backend':label,'preprocess':stats(pre),'inference':stats(inf),'postprocess':stats(post),'total':stats(total)};print(label,result['total']);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--split',default='validation');p.add_argument('--frames',type=int,default=80);p.add_argument('--warmup',type=int,default=10);a=p.parse_args();payload=json.loads(a.dataset.read_text());dataset=[x for x in payload['annotations'] if x.get('split')==a.split and x.get('review_status')=='complete' and x.get('label_origin')=='HUMAN_REVIEWED'][:a.frames];images=[cv2.imread(str(a.dataset.parent/x['image'])) for x in dataset];results=[]
 for t in (1,2):results.append(run(f'ort_fp32_{t}t',ort_factory(a.model,t),images,a.warmup))
 for t in (1,2):results.append(run(f'openvino_fp32_{t}t',ov_factory(a.model,t),images,a.warmup))
 out={'dataset':str(a.dataset),'split':a.split,'samples':len(images),'warmup':a.warmup,'results':results};a.model.with_suffix('.timing.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
