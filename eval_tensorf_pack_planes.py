#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import cv2

from models.recon_utils import (
    DCVC_ALIGN,
    normalize_planes,
    pack_planes_to_rgb,
)

"""
python eval_tensorf_pack_planes.py \
  --den_packing_mode flatten --den_quant_mode global --den_global_range -25 25 --den_r 4 --den_c 4 \
  --app_packing_mode flatten --app_quant_mode global --app_global_range -5 5 --app_r 6 --app_c 8 \
  --emit_png --emit_jpeg --jpeg_quality 80 \
  --ckpt log_2/nerf_chair/chair_codec_compression_34999.th \
  --outdir log_2/nerf_chair/planes_eval 

python eval_tensorf_pack_planes.py \
  --den_packing_mode flat4 --den_quant_mode global --den_global_range -25 25 --den_r 4 --den_c 4 \
  --app_packing_mode mosaic --app_quant_mode global --app_global_range -5 5 --app_r 6 --app_c 8 \
  --emit_png\
  --ckpt log_2/noise_gaussian_heavy/noise_gaussian_heavy_compression_1199.th
"""



def _mkdir(p): os.makedirs(p, exist_ok=True)

def _u8_from_f01(x):
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _write_gray(path, f01_hw):
    u8 = _u8_from_f01(f01_hw)
    ok = cv2.imwrite(path, u8)
    if not ok: raise RuntimeError(f"cv2.imwrite failed: {path}")

def _write_color(path, f01_hw3):
    u8 = _u8_from_f01(f01_hw3[..., ::-1])  # RGB->BGR
    ok = cv2.imwrite(path, u8)
    if not ok: raise RuntimeError(f"cv2.imwrite failed: {path}")

def _write_jpeg_gray(path, f01_hw, quality: int):
    u8 = _u8_from_f01(f01_hw)
    ok = cv2.imwrite(path, u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok: raise RuntimeError(f"cv2.imwrite(.jpg, gray) failed: {path}")

def _write_jpeg_color(path, f01_hw3, quality: int):
    u8 = _u8_from_f01(f01_hw3[..., ::-1])  # RGB->BGR
    ok = cv2.imwrite(path, u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok: raise RuntimeError(f"cv2.imwrite(.jpg, color) failed: {path}")

def _plane_keys(sd: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [k for k in sd.keys() if k.startswith(prefix)]
    def _idx(k):
        try: return int(k.split('.')[1])
        except Exception: return 0
    return sorted(keys, key=_idx)

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='Path to a TensoRF checkpoint (.pth/.tar)')
    ap.add_argument('--align', type=int, default=DCVC_ALIGN)

    # density packing/quant
    ap.add_argument('--den_packing_mode', default='flatten', choices=['flatten', 'mosaic', 'flat4'])
    ap.add_argument('--den_quant_mode',   default='global',  choices=['global','per_channel'])
    ap.add_argument('--den_global_range', nargs=2, type=float, default=[-25.0, 25.0])
    ap.add_argument('--den_r', type=int, default=4)
    ap.add_argument('--den_c', type=int, default=4)

    # appearance packing/quant
    ap.add_argument('--app_packing_mode', default='flatten', choices=['flatten', 'mosaic', 'flat4'])
    ap.add_argument('--app_quant_mode',   default='global',  choices=['global','per_channel'])
    ap.add_argument('--app_global_range', nargs=2, type=float, default=[-5.0, 5.0])
    ap.add_argument('--app_r', type=int, default=6)
    ap.add_argument('--app_c', type=int, default=8)

    # what to emit
    ap.add_argument('--emit_png',  action='store_true')
    ap.add_argument('--emit_jpeg', action='store_true')
    ap.add_argument('--jpeg_quality', type=int, default=80)

    return ap.parse_args()

def _validate_channels(C: int, mode: str, r: int, c: int, plane_name: str):
    if mode == 'flatten':
        if r * c != C:
            raise ValueError(f"{plane_name}: pack(flatten) requires r*c==C (got r*c={r*c}, C={C})")
    elif mode in ('mosaic', 'flat4'):
        if C % 3 != 0:
            raise ValueError(f"{plane_name}: pack({mode}) requires C%3==0 (got C={C})")
        Cpc = C // 3
        s = int(np.sqrt(Cpc))
        if s * s != Cpc:
            raise ValueError(f"{plane_name}: pack({mode}) requires (C/3) to be a perfect square; got C/3={Cpc}")

def _pack_group(sd, keys, base_out, packing_mode, quant_mode, global_range, r, c, align, args):
    meta_planes = {}
    _mkdir(base_out)

    for k in keys:
        x = sd[k].detach().clone()            # [1,C,H,W]
        assert x.dim() == 4 and x.shape[0] == 1, f"{k}: expect [1,C,H,W]"
        _, C, H, W = x.shape

        _validate_channels(C, packing_mode, r, c, k)

        # normalize as in STE
        x01, cmin, scale = normalize_planes(x, mode=quant_mode, global_range=tuple(global_range))

        # pack to canvas
        y_pad, (Hp, Wp) = pack_planes_to_rgb(x01, align=align, mode=packing_mode, r=r, c=c)  # [1,3,Hpad,Wpad]
        rgb01 = y_pad[0].permute(1,2,0).contiguous().cpu().numpy()  # HxWx3 RGB

        # filenames
        stem = k.replace('.', '_')
        is_mono = (packing_mode == 'flatten')
        paths = {}

        if is_mono:
            mono = rgb01[..., 0]  # all channels equal
            if args.emit_png:
                p = os.path.join(base_out, f'{stem}.png')
                _write_gray(p, mono); paths['png'] = p
            if args.emit_jpeg:
                p = os.path.join(base_out, f'{stem}.jpg')
                _write_jpeg_gray(p, mono, args.jpeg_quality); paths['jpg'] = p
        else:
            if args.emit_png:
                p = os.path.join(base_out, f'{stem}.png')
                _write_color(p, rgb01); paths['png'] = p
            if args.emit_jpeg:
                p = os.path.join(base_out, f'{stem}.jpg')
                _write_jpeg_color(p, rgb01, args.jpeg_quality); paths['jpg'] = p

        # bounds for dequant (per-channel)
        cmin_np  = cmin.squeeze().reshape(-1).cpu().numpy()
        cmax_np  = (cmin + scale).squeeze().reshape(-1).cpu().numpy()
        bounds   = np.stack([cmin_np, cmax_np], axis=1).tolist()

        meta_planes[k] = {
            'shape': [int(d) for d in x.shape],             # [1,C,H,W]
            'packing_mode': packing_mode,
            'quant_mode':   quant_mode,
            'global_range': [float(global_range[0]), float(global_range[1])],
            'r': int(r), 'c': int(c),
            'align': int(align),
            'orig_hw': [int(Hp), int(Wp)],                  # before pad
            'stored': paths,                                # {'png': ..., 'jpg': ...}
            'is_mono': bool(is_mono),
            'bounds': bounds,
        }

    return meta_planes

if __name__ == '__main__':
    args = parse_args()
    outdir = Path(args.ckpt).parent / f'planes_eval_{args.app_packing_mode}_{args.den_packing_mode}'
    _mkdir(outdir)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    den_keys = _plane_keys(sd, 'density_plane.')
    app_keys = _plane_keys(sd, 'app_plane.')

    den_meta = _pack_group(
        sd, den_keys, os.path.join(outdir, 'density'),
        args.den_packing_mode, args.den_quant_mode, args.den_global_range,
        args.den_r, args.den_c, args.align, args
    )
    app_meta = _pack_group(
        sd, app_keys, os.path.join(outdir, 'appearance'),
        args.app_packing_mode, args.app_quant_mode, args.app_global_range,
        args.app_r, args.app_c, args.align, args
    )

    meta = {
        'align': int(args.align),
        'density': den_meta,
        'appearance': app_meta,
        'jpeg_quality': int(args.jpeg_quality),
        'emit_png': bool(args.emit_png),
        'emit_jpeg': bool(args.emit_jpeg),
        'note': 'Produced by eval_tensorf_pack_planes.py; supports flatten/mosaic/flat4.',
    }
    with open(os.path.join(outdir, 'planes_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    torch.save(meta, os.path.join(outdir, 'planes_meta.pt'))
    print(f"[DONE] Wrote plane canvases + meta to {outdir}")
