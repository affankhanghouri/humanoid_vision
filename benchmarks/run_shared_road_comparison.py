"""Full-video A/B/B/A replay, serially; no road activation in app defaults."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    out=Path('validation/shared_road');out.mkdir(parents=True,exist_ok=True)
    for name,road in [('A1',False),('B1',True),('B2',True),('A2',False)]:
        cmd=[sys.executable,'benchmarks/benchmark_shared_road.py','--headless','--seconds','0','--output',str(out/f'{name}.json')]
        if road:cmd+=['--road-model','models/road_nano_640.onnx']
        print('START',name,flush=True)
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/f'{name}.json').read_text())
        print('DONE',name,{k:result[k] for k in ['duration_seconds','detector_hz','road_hz','display_fps','stale_percent','worker_utilization_percent']},flush=True)

if __name__=='__main__':main()
