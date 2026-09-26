"""CPU session setup for the pinned Ultralytics ONNX backend."""
from pathlib import Path


def configure_cpu_onnx(model, model_path: str, threads: int) -> None:
    """Let idle detection/pose pools sleep so they do not compete with rendering.

    Ultralytics 8.4.143 does not forward session options through YOLO.predict.
    Its prediction-start callback runs after backend setup and before inference.
    Replace that CPU session once, retaining Ultralytics preprocessing/postprocessing.
    Subsequent warmup runs and worker calls all use the configured session.
    """
    if Path(model_path).suffix.lower() != '.onnx':
        return
    if threads < 1:
        raise ValueError('ONNX thread count must be positive')
    configured = False

    def configure(predictor):
        nonlocal configured
        # Ultralytics CPU setup resets PyTorch's thread count. Its small tensor
        # preprocessing/postprocessing operations do not need another CPU pool
        # competing with ONNX and the GUI. Apply on the actual inference thread.
        import torch
        if predictor.device.type == 'cpu':
            torch.set_num_threads(1)
        if configured:
            return
        backend = predictor.model.backend
        if backend.session.get_providers() != ['CPUExecutionProvider']:
            # GPU sessions have additional bindings and are owned by Ultralytics.
            configured = True
            return
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        backend.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        configured = True

    model.add_callback('on_predict_start', configure)
