"""
Export a trained FullModel checkpoint to ONNX.

Usage (PowerShell):
python d:\Project\src\export_onnx.py --ckpt d:\Project\checkpoints\epoch_50.pt --out d:\Project\checkpoints\model_epoch50.onnx --seq-len 40 --spatial-size 88

Notes:
- The script loads the checkpoint (accepts dict with 'state_dict' or a raw state_dict).
- It attempts to infer num_classes from the checkpoint (backend.fc.weight shape).
- Exports two outputs: 'logits' (B, num_classes) and 'logits_t' (B, T', num_classes).
- Uses dynamic axes for batch and time dimensions so the ONNX can accept variable batch/time sizes.
"""
import os
import argparse
import torch
import sys

# ensure project root for importing src.model
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FullModel


def infer_num_classes_from_ckpt(ckpt):
    # ckpt may be a dict with 'state_dict' or a raw state_dict
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # find backend.fc.weight key
    for k, v in sd.items():
        if k.endswith('backend.fc.weight') or k.endswith('backend.fc.weight'.replace('.', '_')):
            return v.shape[0]
    # fallback: search for 'fc.weight'
    for k, v in sd.items():
        if k.endswith('fc.weight'):
            return v.shape[0]
    raise RuntimeError('Could not infer num_classes from checkpoint state_dict')


def load_state_dict_into_model(model, ckpt):
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # allow keys prefixed with 'module.' from DataParallel
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_sd[nk] = v
    model.load_state_dict(new_sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint (.pt) to load')
    parser.add_argument('--out', required=True, help='Output ONNX path')
    parser.add_argument('--seq-len', type=int, default=40, help='Temporal length for dummy input')
    parser.add_argument('--spatial-size', type=int, default=88, help='Spatial H/W for dummy input')
    parser.add_argument('--in-channels', type=int, default=1, help='Input channels (default 1)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print('Checkpoint not found:', args.ckpt)
        return

    ckpt = torch.load(args.ckpt, map_location='cpu')
    try:
        num_classes = infer_num_classes_from_ckpt(ckpt)
    except Exception as e:
        print('Failed to infer num_classes from checkpoint:', e)
        print('Please re-run with --num-classes (not implemented).')
        raise

    print(f'Inferred num_classes={num_classes} from checkpoint')

    model = FullModel(num_classes=num_classes, in_channels=args.in_channels, frontend_out=64, pretrained_resnet=False, dropout=0.0)
    load_state_dict_into_model(model, ckpt)
    model.eval()
    model.to('cpu')

    # dummy input: (B, T, C, H, W)
    B = 1
    T = args.seq_len
    C = args.in_channels
    H = args.spatial_size
    W = args.spatial_size
    dummy = torch.zeros((B, T, C, H, W), dtype=torch.float)

    # export
    output_path = args.out
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_names = ['input']
    output_names = ['logits', 'logits_t']
    dynamic_axes = {
        'input': {0: 'batch', 1: 'time'},
        'logits': {0: 'batch'},
        'logits_t': {0: 'batch', 1: 'time_out'}
    }

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                (dummy,),
                output_path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            print('Exported ONNX to', output_path)
        except Exception as e:
            print('ONNX export failed:', repr(e))
            raise


if __name__ == '__main__':
    main()
