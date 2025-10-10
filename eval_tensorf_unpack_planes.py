#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import cv2

from models.recon_utils import (
    unpack_rgb_to_planes,
)

def _f01_from_u8(xu8): return xu8.astype(np.float32) / 255.0

def _read_gray(path) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: raise FileNotFoundError(path)
    return _f01_from_u8(im)  # HxW float01

def _read_color_rgb01(path) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return _f01_from_u8(im)  # HxWx3 float01

def _plane_keys(sd: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [k for k in sd.keys() if k.startswith(prefix)]
    def _idx(k):
        try: return int(k.split('.')[1])
        except Exception: return 0
    return sorted(keys, key=_idx)

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='Original TensoRF checkpoint (.pth/.tar) to patch')
    ap.add_argument('--planes_dir', default=None, help='Directory that contains planes_eval/planes_meta.json (default: <ckpt_dir>/planes_eval)')
    ap.add_argument('--prefer', default='jpg,png', help='Preferred extensions to load (comma list)')
    ap.add_argument('--out', default=None, help='Output checkpoint path (default: <ckpt_stem>_recon_planes.pth)')
    return ap.parse_args()

def _choose_path(stored: Dict[str,str], prefer_order: List[str]) -> str:
    for ext in prefer_order:
        if ext in stored: return stored[ext]
    raise FileNotFoundError(f"No stored image among {prefer_order} in {stored.keys()}")

def _reconstruct_group(sd, meta_group, prefer_exts):
    """
    meta_group: meta['density'] or meta['appearance'] (dict per plane key)
    Returns dict: { plane_key: torch.Tensor[1,C,H,W] float32 }
    """
    out = {}
    for k, info in meta_group.items():
        C = int(info['shape'][1])
        H_pad, W_pad = None, None
        Hp, Wp = info['orig_hw']          # before pad
        mode = info['packing_mode']
        r, c  = int(info['r']), int(info['c'])
        is_mono = bool(info['is_mono'])
        bounds = np.asarray(info['bounds'], dtype=np.float32)  # [C,2] lo/hi

        path = _choose_path(info['stored'], prefer_exts)

        if is_mono:
            mono = _read_gray(path)       # Hpad x Wpad (float01)
            H_pad, W_pad = mono.shape
            rgb = np.stack([mono, mono, mono], axis=-1)   # Hpad x Wpad x 3
        else:
            rgb = _read_color_rgb01(path)                 # Hpad x Wpad x 3
            H_pad, W_pad = rgb.shape[:2]

        y = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).contiguous()  # [1,3,Hpad,Wpad]

        # crop to orig (Hp,Wp) then unpack -> [1,C,H,W] in [0,1]
        y_crop = y[..., :Hp, :Wp]
        x01 = unpack_rgb_to_planes(y_crop, C=C, orig_size=(Hp, Wp), mode=mode, r=r, c=c)

        # de-normalize using per-channel bounds
        lo = torch.from_numpy(bounds[:,0]).view(1, C, 1, 1)
        hi = torch.from_numpy(bounds[:,1]).view(1, C, 1, 1)
        rec = (x01 * (hi - lo) + lo).to(torch.float32)

        # sanity to original spatial size from meta['shape']
        _, _, H, W = info['shape']
        if rec.shape[-2:] != (H, W):
            # in practice they should match; if not, center crop
            rec = rec[..., :H, :W]
        out[k] = rec
    return out

if __name__ == '__main__':
    args = parse_args()
    prefer_exts = [e.strip().lower() for e in args.prefer.split(',') if e.strip()]
    planes_dir = Path(args.planes_dir) if args.planes_dir else (Path(args.ckpt).parent / 'planes_eval')
    meta_path = planes_dir / 'planes_meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt).copy()

    # reconstruct planes from images
    den_rec = _reconstruct_group(sd, meta['density'], prefer_exts)
    app_rec = _reconstruct_group(sd, meta['appearance'], prefer_exts)

    # write back into state dict
    for k, v in {**den_rec, **app_rec}.items():
        sd[k] = v

    # save output
    out_path = args.out
    if out_path is None:
        stem = Path(args.ckpt).with_suffix('').name
        out_path = str(Path(args.ckpt).parent / f"{stem}_recon_planes.pth")

    # keep original container structure
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = sd
        torch.save(ckpt, out_path)
    else:
        torch.save(sd, out_path)

    print(f"[DONE] Wrote reconstructed checkpoint to {out_path}")
