"""
Export a trained FullModel checkpoint to TensorFlow Lite (.tflite).

Workflow:
  PyTorch checkpoint (.pt) -> ONNX -> TensorFlow SavedModel -> TFLite

Requirements (install in the same Python environment you run this script):
  pip install onnx onnxruntime onnx-tf tensorflow

Usage (PowerShell):
  python d:\Project\src\export_tflite.py --ckpt D:\Project\src\checkpoints\final_model.pt --out d:\Project\src\checkpoints\final.tflite

Notes:
- This script creates a temporary ONNX and SavedModel in a temp directory by default.
- You can adjust --seq-len, --spatial-size and --in-channels to match model input used during training.
- If conversion fails, install/verify versions of onnx / onnx-tf / tensorflow compatible with your environment.
"""

import os
import argparse
import tempfile
import shutil
import torch
import sys
import subprocess
import shutil

# ensure project root for importing src.model
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FullModel


def infer_num_classes_from_ckpt(ckpt):
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    for k, v in sd.items():
        if k.endswith('backend.fc.weight') or k.endswith('backend_fc_weight'):
            return v.shape[0]
    for k, v in sd.items():
        if k.endswith('fc.weight'):
            return v.shape[0]
    raise RuntimeError('Could not infer num_classes from checkpoint state_dict')


def load_state_dict_into_model(model, ckpt):
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_sd[nk] = v
    model.load_state_dict(new_sd)


def check_imports(mode='default'):
    """Return list of missing packages depending on conversion mode.
    mode: 'default' (onnx-tf + tensorflow), 'ai-edge', 'litert'
    """
    missing = []
    try:
        import onnx  # noqa: F401
    except Exception:
        missing.append('onnx')

    if mode == 'default':
        try:
            import onnx_tf  # noqa: F401
        except Exception:
            missing.append('onnx-tf')
        try:
            import tensorflow as tf  # noqa: F401
        except Exception:
            missing.append('tensorflow')
    elif mode == 'ai-edge':
        # ai-edge-torch will be checked at conversion time
        pass
    elif mode == 'litert':
        # litert-torch will be checked at conversion time
        pass
    return missing


def export_to_onnx(model, dummy, onnx_path, opset=12):
    input_names = ['input']
    output_names = ['logits', 'logits_t']
    dynamic_axes = {
        'input': {0: 'batch', 1: 'time'},
        'logits': {0: 'batch'},
        'logits_t': {0: 'batch', 1: 'time_out'}
    }
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def onnx_to_saved_model(onnx_path, saved_model_dir):
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    # export SavedModel
    tf_rep.export_graph(saved_model_dir)


def ai_edge_torch_convert(onnx_path, tflite_path):
    """
    Try to convert ONNX -> TFLite using ai-edge-torch. This will:
    - Try to import python package `ai_edge_torch` and call a conversion function if available.
    - Otherwise try to call the `ai-edge-torch` CLI if on PATH.

    The exact API of ai-edge-torch may vary; this function attempts reasonable fallbacks.
    """
    # First, try python package import
    try:
        import ai_edge_torch
        # try common function names
        if hasattr(ai_edge_torch, 'convert_onnx_to_tflite'):
            return ai_edge_torch.convert_onnx_to_tflite(onnx_path, tflite_path)
        if hasattr(ai_edge_torch, 'convert'):
            return ai_edge_torch.convert(input=onnx_path, output=tflite_path)
        # unknown API
        raise RuntimeError('ai_edge_torch python package found but no known convert API')
    except Exception:
        # try CLI
        cli = shutil.which('ai-edge-torch') or shutil.which('ai_edge_torch')
        if cli:
            # attempt common CLI forms
            cmds = [
                [cli, 'convert', '--onnx', onnx_path, '--out', tflite_path],
                [cli, 'convert', '--input', onnx_path, '--output', tflite_path],
                [cli, 'convert', onnx_path, tflite_path],
            ]
            for cmd in cmds:
                try:
                    subprocess.run(cmd, check=True)
                    return
                except Exception:
                    pass
            raise RuntimeError('ai-edge-torch CLI found but conversion commands failed')
        else:
            raise RuntimeError('ai-edge-torch not installed (python package or CLI not found)')


def litert_torch_convert(onnx_path, tflite_path):
    """
    Convert ONNX -> TFLite using litert-torch (google-ai-edge/litert-torch).
    Try python package API first, then CLI fallbacks.
    """
    try:
        import litert_torch
        # try common API
        if hasattr(litert_torch, 'convert_onnx_to_tflite'):
            return litert_torch.convert_onnx_to_tflite(onnx_path, tflite_path)
        if hasattr(litert_torch, 'convert'):
            return litert_torch.convert(input=onnx_path, output=tflite_path)
        raise RuntimeError('litert_torch python package found but no known convert API')
    except Exception:
        cli = shutil.which('litert-torch') or shutil.which('litert_torch') or shutil.which('lt-torch')
        if cli:
            cmds = [
                [cli, 'convert', '--onnx', onnx_path, '--out', tflite_path],
                [cli, 'convert', '--input', onnx_path, '--output', tflite_path],
                [cli, onnx_path, tflite_path],
            ]
            for cmd in cmds:
                try:
                    subprocess.run(cmd, check=True)
                    return
                except Exception:
                    pass
            raise RuntimeError('litert-torch CLI found but conversion commands failed')
        else:
            raise RuntimeError('litert-torch not installed (python package or CLI not found)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--out', required=True, help='Output .tflite path')
    parser.add_argument('--seq-len', type=int, default=40, help='Temporal length for dummy input')
    parser.add_argument('--spatial-size', type=int, default=88, help='Spatial H/W for dummy input')
    parser.add_argument('--in-channels', type=int, default=1, help='Input channels')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary ONNX and SavedModel files')
    parser.add_argument('--optimize', action='store_true', help='Enable TFLite optimizations')
    parser.add_argument('--use-ai-edge-torch', action='store_true', help='Use ai-edge-torch for ONNX->TFLite conversion if available')
    parser.add_argument('--use-litert-torch', action='store_true', help='Use litert-torch (google-ai-edge/litert-torch) for ONNX->TFLite conversion')
    args = parser.parse_args()

    # determine required imports based on chosen conversion method
    mode = 'default'
    if args.use_ai_edge_torch:
        mode = 'ai-edge'
    if args.use_litert_torch:
        mode = 'litert'

    missing = check_imports(mode=mode)
    if missing:
        print('Missing required packages for conversion:', ', '.join(missing))
        if mode == 'default':
            print('Install them: pip install onnx onnx-tf tensorflow')
        else:
            print('Install onnx and the converter of your choice (ai-edge-torch or litert-torch).')
        return

    if not os.path.isfile(args.ckpt):
        print('Checkpoint not found:', args.ckpt)
        return

    tmpdir = tempfile.mkdtemp(prefix='export_tflite_')
    onnx_path = os.path.join(tmpdir, 'model.onnx')
    saved_model_dir = os.path.join(tmpdir, 'saved_model')
    tflite_out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(tflite_out), exist_ok=True)

    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        num_classes = infer_num_classes_from_ckpt(ckpt)
        print('Inferred num_classes =', num_classes)

        model = FullModel(num_classes=num_classes, in_channels=args.in_channels, frontend_out=64, pretrained_resnet=False, dropout=0.0)
        load_state_dict_into_model(model, ckpt)
        model.eval()
        model.to('cpu')

        B = 1
        T = args.seq_len
        C = args.in_channels
        H = args.spatial_size
        W = args.spatial_size
        dummy = torch.zeros((B, T, C, H, W), dtype=torch.float)

        print('Exporting to ONNX:', onnx_path)
        export_to_onnx(model, dummy, onnx_path, opset=args.opset)

        if args.use_ai_edge_torch:
            print('Converting ONNX -> TFLite via ai-edge-torch:', tflite_out)
            ai_edge_torch_convert(onnx_path, tflite_out)
        elif args.use_litert_torch:
            print('Converting ONNX -> TFLite via litert-torch:', tflite_out)
            litert_torch_convert(onnx_path, tflite_out)
        else:
            print('Converting ONNX -> TensorFlow SavedModel')
            onnx_to_saved_model(onnx_path, saved_model_dir)

            print('Converting SavedModel -> TFLite:', tflite_out)
            saved_model_to_tflite(saved_model_dir, tflite_out, optimize=args.optimize)

        print('TFLite export completed:', tflite_out)
        if args.keep_temp:
            print('Temporary files kept at', tmpdir)
        else:
            shutil.rmtree(tmpdir)
    except Exception as e:
        print('Conversion failed:', repr(e))
        print('Temporary files left at:', tmpdir)


if __name__ == '__main__':
    main()
