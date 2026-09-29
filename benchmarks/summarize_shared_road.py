"""Retain concise shared-worker evidence without committing replay artifacts."""
import json
from pathlib import Path


def main():
    root=Path('validation/shared_road')
    runs={}
    # B3/B4 repeat the road-enabled legs with explicit preprocessing and
    # postprocessing timing. Keep A1/A2 as the bracketing baselines.
    for name in ('A1','B3_stages','B4_stages','A2'):
        data=json.loads((root/f'{name}.json').read_text())
        runs[name]={k:v for k,v in data.items() if k not in ('dispatches','task_intervals','measurement_start','measurement_end')}
        roads=[x for x in data['dispatches'] if x['kind']=='road']
        runs[name]['road_dispatch_input_age_max_ms']=max(((x['start']-x['source'])*1000 for x in roads),default=None)
        intervals=data['task_intervals']
        runs[name]['overlapping_heavy_intervals']=sum(a[1]>b[0]+.001 for a,b in zip(intervals,intervals[1:]))
    output=Path('benchmarks/road_study/shared_worker_summary.json')
    output.write_text(json.dumps(runs,indent=2)+'\n')
    for name,d in runs.items():
        print(name, {k:d[k] for k in ('detector_hz','road_hz','display_fps','stale_percent','worker_utilization_percent','process_cpu_machine_percent')})
        print('  detector source age:',d['state_age'])
        print('  task latency:',d['task_total'])

if __name__=='__main__':main()
