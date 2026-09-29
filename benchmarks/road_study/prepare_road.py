"""Prepare ignored experimental road models; requires local original Nano source.

Example: python benchmarks/road_study/prepare_road.py --source /tmp/tlnp
No network downloads or production-model overwrite.
"""
import argparse
from pathlib import Path
import sys
import onnx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='TwinLiteNetPlus source checkout, needed for reduced export')
    args=parser.parse_args()
    base=Path('models/twinlitenetplus_nano.onnx')
    onnx.utils.extract_model(str(base),'models/road_nano_640.onnx',['images'],['drivable_area'])
    if args.source:
        import torch
        from argparse import Namespace
        torch.set_num_threads(1)
        sys.path.insert(0,str(args.source.resolve()))
        from model.model import TwinLiteNetPlus
        model=TwinLiteNetPlus(Namespace(config='nano'))
        weights=torch.load('models/twinlitenetplus_nano.pth',map_location='cpu',weights_only=True)
        if 'state_dict' in weights:weights=weights['state_dict']
        model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
        model.eval()
        target='models/road_nano_320_both.onnx'
        with torch.no_grad():
            torch.onnx.export(model,torch.zeros(1,3,192,320),target,input_names=['images'],
                              output_names=['drivable_area','lane_line'],opset_version=17,dynamo=False)
        onnx.utils.extract_model(target,'models/road_nano_320.onnx',['images'],['drivable_area'])
    for path in Path('models').glob('road_nano_*.onnx'):
        onnx.checker.check_model(str(path));print(path,path.stat().st_size)

if __name__=='__main__':main()
