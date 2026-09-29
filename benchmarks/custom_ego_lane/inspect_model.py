#!/usr/bin/env python3
import json,torch
from model import INPUT_SIZE,TinyEgoLaneNet,parameter_count
m=TinyEgoLaneNet(False).eval();macs=0
def hook(module,args,out):
 global macs
 x=args[0]
 if isinstance(module,torch.nn.Conv2d):macs+=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
 elif isinstance(module,torch.nn.Linear):macs+=out.numel()*module.in_features
handles=[x.register_forward_hook(hook) for x in m.modules() if isinstance(x,(torch.nn.Conv2d,torch.nn.Linear))]
with torch.no_grad():outputs=m(torch.zeros(1,3,*INPUT_SIZE))
for h in handles:h.remove()
print(json.dumps({'parameters':parameter_count(m),'macs':macs,'approx_flops':2*macs,'input':[1,3,*INPUT_SIZE],'outputs':[list(x.shape) for x in outputs]},indent=2))
