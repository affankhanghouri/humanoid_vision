"""Export the original LSTR checkpoint on CPU without its training dependencies.

Use a local checkout of https://github.com/liuruijin17/LSTR (BSD-3-Clause):
python benchmarks/road_study/export_lstr.py --source /path/to/LSTR
The source is copied to a temporary directory; the original stays unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import cv2
import numpy as np
import torch
import onnx
import onnxruntime as ort
from benchmark import lane_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--video',default='test_data/test_video.mp4')
    args=parser.parse_args()
    torch.set_num_threads(1);cv2.setNumThreads(1)
    source=args.source.resolve()
    with tempfile.TemporaryDirectory(prefix='lstr_export_') as directory:
        temp=Path(directory)
        files=['config.py','config/LSTR.json','models/LSTR.py','models/py_utils/kp.py',
               'models/py_utils/transformer.py','models/py_utils/position_encoding.py','LICENSE']
        hashes={}
        for name in files:
            target=temp/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
            hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        (temp/'models/__init__.py').write_text('')
        (temp/'models/py_utils/__init__.py').write_text('from .kp import kp\n')
        # Remove only unreachable training/debug code and its imports. Inference stays intact.
        p=temp/'models/py_utils/kp.py';text=p.read_text().split('class AELoss')[0]
        for line in ['from .detr_loss import SetCriterion','from .matcher import build_matcher',
                     'from .misc import *','from sample.vis import save_debug_images_boxes']:
            text=text.replace(line,'')
        p.write_text(text)
        p=temp/'models/LSTR.py'
        p.write_text(p.read_text().split('class loss(AELoss)')[0].replace('from .py_utils import kp, AELoss','from .py_utils import kp'))
        sys.path.insert(0,str(temp))
        from config import system_configs
        from models.LSTR import model
        system_configs.update_config(json.loads((temp/'config/LSTR.json').read_text())['system'])
        network=model().eval()
        weights_path=source/'cache/nnet/LSTR/LSTR_500000.pkl'
        weights=torch.load(weights_path,map_location='cpu',weights_only=True)
        # THOP profiling counters are not learned weights or inference buffers.
        weights={k.removeprefix('module.'):v for k,v in weights.items()
                 if k.rsplit('.',1)[-1] not in {'total_ops','total_params'}}
        network.load_state_dict(weights,strict=True)
        class Export(torch.nn.Module):
            def __init__(self):super().__init__();self.net=network
            def forward(self,image,mask):
                result,_=self.net(image,mask)
                return result['pred_logits'],result['pred_curves']
        wrapper=Export().eval()
        result={'checkpoint_sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                'source_hashes':hashes,'parameters':sum(p.numel() for p in network.parameters()),'exports':[]}
        for h,w in [(360,640),(180,320)]:
            path=Path(f'models/lstr_{w}.onnx')
            inputs=(torch.zeros(1,3,h,w),torch.zeros(1,1,h,w))
            with torch.no_grad():
                torch.onnx.export(wrapper,inputs,str(path),input_names=['image','mask'],
                                  output_names=['pred_logits','pred_curves'],opset_version=17,dynamo=False)
            onnx.checker.check_model(str(path))
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            errors=[];cap=cv2.VideoCapture(args.video)
            for frame_id in [100,919,1738]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,frame=cap.read()
                if not ok:raise RuntimeError('Missing export validation video frame')
                image=lane_input(frame,h,w);mask=np.zeros((1,1,h,w),np.float32)
                with torch.no_grad():expected=wrapper(torch.from_numpy(image),torch.from_numpy(mask))
                actual=session.run(None,{'image':image,'mask':mask})
                for a,b in zip(actual,expected):
                    b=b.numpy();np.testing.assert_allclose(a,b,rtol=1e-3,atol=1e-4)
                    errors.append(float(np.max(np.abs(a-b))))
            cap.release()
            result['exports'].append({'path':str(path),'bytes':path.stat().st_size,
                                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'max_abs_error':max(errors)})
            print(result['exports'][-1],flush=True)
        out=Path('validation/road_study/export_lstr.json');out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
