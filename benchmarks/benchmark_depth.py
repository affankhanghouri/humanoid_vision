"""Benchmark two small relative-depth models on CPU with OpenVINO.

    python benchmarks/benchmark_depth.py --download --frames 20

Models: MiDaS v2.1 Small (256 square), Depth Anything V2 Small (266 square).
The latter uses a reduced resolution for this CPU speed study; use
--depth-anything-size 518 to measure its standard nominal resolution. Both use
square resizing here. Timing does not measure depth quality or metric accuracy.
Outputs are relative depth/inverse-depth scores, not calibrated meters.

Download/export/compile/warm-up and video decoding are excluded from frame timing.
Total time includes RGB resize/normalization, inference, and output upsampling.
Each model is measured in a fresh process so PyTorch export memory and the other
model do not contaminate memory measurements. RSS includes the Python runtime,
OpenVINO, model, and frame buffers; peak RSS also includes compilation.

Requires existing OpenVINO, NumPy and OpenCV packages. First-time Depth Anything
export additionally needs torch, transformers and safetensors. No runtime
perception module is added. Model downloads require --download; later runs reuse
local artifacts. JSON includes model hashes, versions, input size, timings, Hz,
and Linux RSS. The 1–3 Hz target is standalone speed, not concurrent YOLO speed.

Primary model/preprocessing sources:
https://github.com/isl-org/MiDaS
https://github.com/isl-org/MiDaS/blob/master/midas/model_loader.py
https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/main/preprocessor_config.json
"""
import argparse
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ('midas-small', 'depth-anything-v2-small')
MIDAS_RELEASE = 'https://github.com/isl-org/MiDaS/releases/download/v3_1/'
DEPTH_REPO = 'depth-anything/Depth-Anything-V2-Small-hf'


def model_path(args, name):
    if name == 'midas-small':
        return args.model_dir / 'openvino_midas_v21_small_256.xml'
    return args.model_dir / f'depth_anything_v2_small_{args.depth_anything_size}.xml'


def download(url, destination):
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args, name):
    path = model_path(args, name)
    if path.is_file() and path.with_suffix('.bin').is_file():
        return
    if not args.download:
        raise FileNotFoundError(f'{path} is missing; rerun with --download to fetch/export official weights')
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == 'midas-small':
        for asset in (path, path.with_suffix('.bin')):
            if not asset.is_file():
                download(MIDAS_RELEASE + asset.name, asset)
        return

    import torch
    import openvino as ov
    from transformers import AutoModelForDepthEstimation

    torch.set_num_threads(args.threads)
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_REPO, cache_dir=str(args.model_dir / 'hf'),
        use_safetensors=True, trust_remote_code=False,
    ).eval()

    class DepthOutput(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).predicted_depth

    size = args.depth_anything_size
    with torch.inference_mode():
        converted = ov.convert_model(
            DepthOutput(model), example_input=torch.zeros(1, 3, size, size),
            input=[1, 3, size, size],
        )
    ov.save_model(converted, str(path), compress_to_fp16=True)


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('Linux VmRSS unavailable')


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def measure(args, name):
    import cv2
    import numpy as np
    import openvino as ov

    cv2.setNumThreads(1)
    baseline_rss = rss_mib()
    path = model_path(args, name)
    core = ov.Core()
    compiled = core.compile_model(str(path), 'CPU', {
        'PERFORMANCE_HINT': 'LATENCY',
        'INFERENCE_NUM_THREADS': args.threads,
        'NUM_STREAMS': 1,
    })
    shape = list(compiled.input(0).shape)
    if len(shape) != 4 or shape[:2] != [1, 3]:
        raise ValueError(f'Expected static NCHW RGB model input; got {shape}')
    height, width = shape[2:]
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    request = compiled.create_infer_request()

    def predict(frame):
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        tensor = np.ascontiguousarray(((resized - mean) / std).transpose(2, 0, 1)[None])
        inference_start = time.perf_counter()
        output = request.infer([tensor])[compiled.output(0)]
        inference_end = time.perf_counter()
        depth = np.squeeze(output)
        if depth.ndim != 2:
            raise ValueError(f'Expected 2D depth output; got {depth.shape}')
        depth = cv2.resize(depth, frame.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        end = time.perf_counter()
        if not np.isfinite(depth).all() or float(np.ptp(depth)) <= 0:
            raise ValueError('Depth output is non-finite or constant')
        return (end-start)*1000, (inference_end-inference_start)*1000, depth

    capture = cv2.VideoCapture(str(args.video))
    samples, inference, memory = [], [], []
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Could not open video: {args.video}')
        success, first = capture.read()
        if not success:
            raise RuntimeError('Video contains no readable frames')
        for _ in range(args.warmup):
            predict(first)
        runtime_rss = rss_mib()
        # Reopen so both models measure the same source frames, starting at zero.
        capture.release()
        capture = cv2.VideoCapture(str(args.video))
        for index in range(args.frames):
            success, frame = capture.read()
            if not success:
                break
            total_ms, infer_ms, depth = predict(frame)
            samples.append(total_ms)
            inference.append(infer_ms)
            memory.append(rss_mib())
            print(f'{name} {index+1}/{args.frames}: {total_ms:.1f} ms (infer {infer_ms:.1f} ms)', flush=True)
        if not samples:
            raise RuntimeError('No frames measured')
    finally:
        capture.release()

    def stats(values):
        return {'average_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95))}

    average_ms = float(np.mean(samples))
    return {
        'model': name, 'model_path': str(path),
        'source': 'https://github.com/isl-org/MiDaS' if name == 'midas-small' else f'https://huggingface.co/{DEPTH_REPO}',
        'sha256': {asset.name: file_sha256(asset) for asset in (path, path.with_suffix('.bin'))},
        'device': core.get_property('CPU', 'FULL_DEVICE_NAME'), 'threads': args.threads,
        'input_shape_nchw': shape, 'resize_mode': 'square',
        'depth_units': 'relative scores; not meters',
        'samples': len(samples), 'warmup_runs': args.warmup,
        'total': stats(samples), 'inference': stats(inference),
        'approx_hz': 1000 / average_ms, 'meets_1hz_minimum': average_ms <= 1000,
        'rss_before_compile_mib': baseline_rss, 'rss_after_warmup_mib': runtime_rss,
        'rss_runtime_max_mib': max(memory),
        'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'last_output_range': [float(depth.min()), float(depth.max())],
        'versions': {'python': platform.python_version(), 'openvino': ov.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--video', type=Path, default=ROOT / 'test_data/test_video.mp4')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/depth_benchmark')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/depth_benchmark.json')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--depth-anything-size', type=int, default=266)
    parser.add_argument('--phase', choices=('prepare', 'measure'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.frames, args.warmup, args.threads, args.depth_anything_size) < 1:
        parser.error('frames, warmup, threads, and size must be positive')
    if args.depth_anything_size % 14:
        parser.error('Depth Anything size must be a multiple of 14 (e.g. 266 or 518)')
    if not args.video.is_file():
        parser.error(f'Video not found: {args.video}')
    args.model_dir = args.model_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase:
        name = args.models[0]
        if args.phase == 'prepare':
            prepare(args, name)
        else:
            args.output.write_text(json.dumps(measure(args, name), indent=2) + '\n')
        return

    results = []
    for name in dict.fromkeys(args.models):
        print(f'Preparing {name}...', flush=True)
        child_output = args.output.with_name(f'{args.output.stem}-{name}.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--models', name,
                   '--video', str(args.video.resolve()), '--model-dir', str(args.model_dir),
                   '--frames', str(args.frames), '--warmup', str(args.warmup),
                   '--threads', str(args.threads), '--depth-anything-size', str(args.depth_anything_size),
                   '--output', str(child_output)]
        if args.download:
            command.append('--download')
        try:
            subprocess.run(command + ['--phase', 'prepare'], check=True)
            subprocess.run(command + ['--phase', 'measure'], check=True)
            results.append(json.loads(child_output.read_text()))
        except subprocess.CalledProcessError as exc:
            results.append({'model': name, 'error': f'Benchmark subprocess exited {exc.returncode}; see log above'})
        report = {'platform': platform.platform(), 'video': str(args.video.resolve()),
                  'target_hz': [1, 3], 'results': results,
                  'note': 'Standalone latency only; accuracy and joint YOLO scheduling remain unvalidated.'}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for result in results:
        if 'error' in result:
            print(f"{result['model']}: FAILED ({result['error']})")
        else:
            print(f"{result['model']}: avg={result['total']['average_ms']:.1f} ms "
                  f"p95={result['total']['p95_ms']:.1f} ms "
                  f"{result['approx_hz']:.2f} Hz RSS={result['rss_runtime_max_mib']:.1f} MiB")
    print(f'Report: {args.output}')
    if any('error' in result for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
