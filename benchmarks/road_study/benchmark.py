"""Serial CPU experiments, never imported by the live pipeline.

python benchmarks/road_study/benchmark.py --help
Timings exclude video decoding, drawing, disk I/O and session initialization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import cv2
import numpy as np
import onnxruntime as ort
from geometry import decode_lstr, select_ego


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(("mean", "median", "p95", "p99"), map(float, (a.mean(), np.median(a), np.percentile(a,95), np.percentile(a,99)))))


def road_input(frame, h, w):
    fh, fw = frame.shape[:2]
    scale = min(h/fh, w/fw)
    rh, rw = round(fh*scale), round(fw*scale)
    top, left = (h-rh)//2, (w-rw)//2
    padded = np.full((h,w,3),114,np.uint8)
    padded[top:top+rh,left:left+rw] = cv2.resize(frame,(rw,rh))
    return cv2.dnn.blobFromImage(padded,1/255.,swapRB=True), (left,top,rw,rh)


def lane_input(frame, h, w):
    # Original LSTR test/tusimple.py + db/tusimple.py: BGR, custom mean/std.
    im = cv2.resize(frame,(w,h)).astype(np.float32) / 255.
    im -= np.array([.40789654,.44719302,.47026115],np.float32)
    im /= np.array([.28863828,.27408164,.27809835],np.float32)
    return np.ascontiguousarray(im.transpose(2,0,1)[None])


class Runner:
    def __init__(self, model, backend, threads, optimization):
        start=time.perf_counter()
        if backend == 'ort':
            opts=ort.SessionOptions()
            opts.intra_op_num_threads=threads
            opts.inter_op_num_threads=1
            opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, 'ORT_ENABLE_ALL' if optimization=='all' else 'ORT_DISABLE_ALL')
            opts.add_session_config_entry('session.intra_op.allow_spinning','0')
            opts.add_session_config_entry('session.inter_op.allow_spinning','0')
            self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
            self.names=[i.name for i in self.session.get_inputs()]
            self.outputs=[o.name for o in self.session.get_outputs()]
            self.shapes=[i.shape for i in self.session.get_inputs()]
            self.run=lambda feed:self.session.run(None,feed)
        else:
            import openvino as ov
            core=ov.Core()
            self.session=core.compile_model(str(model),'CPU',{'INFERENCE_NUM_THREADS':threads,'NUM_STREAMS':1,'INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY'})
            self.names=[i.get_any_name() for i in self.session.inputs]
            self.outputs=[o.get_any_name() for o in self.session.outputs]
            self.shapes=[list(i.shape) for i in self.session.inputs]
            self.run=lambda feed:list(self.session(feed).values())
        self.setup_ms=(time.perf_counter()-start)*1000
        self.h,self.w=map(int,self.shapes[0][-2:])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--kind',choices=['road','lane'],required=True)
    ap.add_argument('--backend',choices=['ort','openvino'],default='ort')
    ap.add_argument('--threads',type=int,choices=[1,2],default=1)
    ap.add_argument('--optimization',choices=['all','off'],default='all')
    ap.add_argument('--video',type=Path,default=Path('test_data/test_video.mp4'))
    ap.add_argument('--frames',type=int,default=60)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--road-masks',type=Path,help='Baseline road results directory; required to validate ego candidates')
    ap.add_argument('--compare-road',type=Path,help='Reference road results; computes agreement, NOT accuracy')
    args=ap.parse_args()
    if args.frames < 12: ap.error('--frames must be at least 12')
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    runner=Runner(args.model,args.backend,args.threads,args.optimization)
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {args.video}')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    visual_ids=set(np.linspace(round(count*.05),round(count*.95),12,dtype=int).tolist())
    ids=sorted(set(np.linspace(round(count*.05),round(count*.95),args.frames,dtype=int).tolist())|visual_ids)
    timings={name:[] for name in ['pre','infer','post','total']}; rows=[]; tiles=[]; agreements=[]
    padding={name:np.zeros(shape,np.float32) for name,shape in zip(runner.names[1:],runner.shapes[1:])}
    for index,frame_id in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Cannot read frame {frame_id}')
        road=None
        if args.kind=='lane' and args.road_masks:
            path=args.road_masks/f'mask_{frame_id:05d}.png'
            if path.exists(): road=(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
        start=time.perf_counter()
        if args.kind=='road': tensor,crop=road_input(frame,runner.h,runner.w)
        else: tensor=lane_input(frame,runner.h,runner.w)
        feed={runner.names[0]:tensor,**padding}
        t1=time.perf_counter()
        if index==0:
            for _ in range(5): runner.run(feed)
            start=time.perf_counter()-(t1-start); t1=time.perf_counter()
        outputs=runner.run(feed); t2=time.perf_counter()
        row={'frame_id':frame_id,'source_timestamp':frame_id/fps}
        if args.kind=='road':
            out=outputs[runner.outputs.index('drivable_area')]
            if out.shape != (1,2,runner.h,runner.w) or not np.isfinite(out).all():
                raise ValueError(f'Invalid road output: {out.shape}')
            # Exact binary argmax, including background on ties; no full-image upsample.
            mask=(out[0,1]>out[0,0]).astype(np.uint8)
            x,y,w,h=crop; mask=mask[y:y+h,x:x+w]
            row['road_fraction']=float(mask.mean())
        else:
            by_shape={a.shape[-1]:a for a in outputs}
            lanes=decode_lstr(by_shape[2],by_shape[8])
            row['lanes']=[{'query':l.query,'confidence':l.confidence,'points':l.points.tolist()} for l in lanes]
            row['ego']=select_ego(lanes,road)
        end=time.perf_counter()
        for k,v in zip(timings, [t1-start,t2-t1,end-t2,end-start]):timings[k].append(v*1000)
        rows.append(row)
        if args.kind=='road':
            cv2.imwrite(str(args.out/f'mask_{frame_id:05d}.png'),mask*255)
            if args.compare_road:
                ref=cv2.imread(str(args.compare_road/f'mask_{frame_id:05d}.png'),0)
                if ref is None: raise RuntimeError('Missing reference mask')
                scaled=cv2.resize(mask,(ref.shape[1],ref.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                ref=ref>0; union=np.logical_or(ref,scaled).sum()
                agreements.append(float(np.logical_and(ref,scaled).sum()/union) if union else 1.)
        if frame_id in visual_ids:
            tile=cv2.resize(frame,(576,360))
            if args.kind=='road':
                m=cv2.resize(mask,(576,360),interpolation=cv2.INTER_NEAREST)>0
                tile[m]=(tile[m]*.6+np.array([0,180,0])*.4).astype(np.uint8)
                label=f'road fraction={row["road_fraction"]:.2f}'
            else:
                for lane in lanes:
                    points=np.rint(lane.points*np.array([576,360])).astype(np.int32)
                    cv2.polylines(tile,[points],False,(0,220,255),2)
                    cv2.putText(tile,f'{lane.query}:{lane.confidence:.2f}',tuple(points[-1]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
                ego=row['ego']; label=f'ego={ego["valid"]} {ego["reason"]}'
                if ego['valid']:
                    poly=np.array(ego['left']+ego['right'][::-1])*[576,360]
                    overlay=tile.copy();cv2.fillPoly(overlay,[poly.astype(np.int32)],(30,180,30));tile=cv2.addWeighted(tile,.75,overlay,.25,0)
            cv2.rectangle(tile,(0,0),(576,44),(0,0,0),-1)
            cv2.putText(tile,f'{frame_id}: {label}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(tile,f'{args.model.stem} {args.backend} t{args.threads}',(8,38),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.imwrite(str(args.out/f'frame_{frame_id:05d}.jpg'),tile);tiles.append(tile)
            print(f'[{len(tiles):02}/12] frame={frame_id} {label}',flush=True)
    cap.release()
    montage=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)])
    cv2.imwrite(str(args.out/'montage.jpg'),montage)
    result={'model':str(args.model),'sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'model_bytes':args.model.stat().st_size,'kind':args.kind,'backend':args.backend,'threads':args.threads,
            'optimization':args.optimization,'input_shapes':runner.shapes,'outputs':runner.outputs,
            'setup_ms':runner.setup_ms,'samples':len(rows),'warmup':5,'video':str(args.video),'video_fps':fps,
            'python':platform.python_version(),'ort':ort.__version__,'opencv':cv2.__version__,
            'timing_ms':{k:stats(v) for k,v in timings.items()},'raw_timing_ms':timings,'observations':rows}
    if args.backend=='openvino':
        import openvino as ov
        result['openvino']=ov.__version__
    if agreements: result['mask_agreement_iou']={'mean':float(np.mean(agreements)),'min':float(min(agreements))}
    if args.kind=='lane':result['ego_valid']=sum(r['ego']['valid'] for r in rows)
    (args.out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['observations','raw_timing_ms']},indent=2),flush=True)

if __name__=='__main__':main()
