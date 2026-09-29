"""Run CPU comparisons serially. Prepare the ignored models before invoking.

Each run saves raw stage timings, metadata, predictions and 12 visual samples.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('validation/road_study'))
    parser.add_argument('--frames',type=int,default=60)
    parser.add_argument('--video',default='test_data/test_video.mp4')
    args=parser.parse_args()
    script=Path(__file__).with_name('benchmark.py')
    runs=[('baseline','road','twinlitenetplus_nano.onnx','ort',1,'all')]
    for backend,label in [('ort','t'),('openvino','ov')]:
        for threads in (1,2):
            runs.append((f'road640_{label}{threads}','road','road_nano_640.onnx',backend,threads,'all'))
    for threads in (1,2):
        runs.append((f'road320_t{threads}','road','road_nano_320.onnx','ort',threads,'all'))
    runs.append(('road640_noopt','road','road_nano_640.onnx','ort',1,'off'))
    for width in (640,320):
        for threads in (1,2):
            runs.append((f'lane{width}_t{threads}','lane',f'lstr_{width}.onnx','ort',threads,'all'))
    for threads in (1,2):
        runs.append((f'lane640_ov{threads}','lane','lstr_640.onnx','openvino',threads,'all'))
    runs.extend([('baseline_repeat','road','twinlitenetplus_nano.onnx','ort',1,'all'),
                 ('road640_ov2_repeat','road','road_nano_640.onnx','openvino',2,'all')])
    for name,kind,model,backend,threads,optimization in runs:
        out=args.out/name;out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(script),'--model',f'models/{model}','--kind',kind,
                 '--backend',backend,'--threads',str(threads),'--optimization',optimization,
                 '--video',args.video,'--frames',str(args.frames),'--out',str(out)]
        if name!='baseline':command += ['--road-masks' if kind=='lane' else '--compare-road',str(args.out/'baseline')]
        print('Running',name,flush=True)
        with (out/'run.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Saved',out/'results.json',flush=True)

if __name__=='__main__':main()
