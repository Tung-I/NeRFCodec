#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import cv2
"""
Usage:

"""
from models.recon_utils import (
    DCVC_ALIGN,
    normalize_planes,
    pack_planes_to_rgb,
    unpack_rgb_to_planes,
    jpeg_roundtrip_mono,
    jpeg_roundtrip_color,
)


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _plane_keys(sd: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [k for k in sd.keys() if k.startswith(prefix)]
    def _idx(k):
        try: return int(k.split('.')[1])
        except Exception: return 0
    return sorted(keys, key=_idx)

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

def _to_u8_from_f01(x):
    x = np.clip(x, 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)

def _write_gray(path, f01_hw):
    ok = cv2.imwrite(path, _to_u8_from_f01(f01_hw))
    if not ok: raise RuntimeError(f"cv2.imwrite failed: {path}")

def _write_color(path, f01_hw3_rgb):
    # OpenCV expects BGR on write
    ok = cv2.imwrite(path, _to_u8_from_f01(f01_hw3_rgb[..., ::-1]))
    if not ok: raise RuntimeError(f"cv2.imwrite failed: {path}")


# ------------------------------------------------------------
# Core per-plane in-memory JPEG reconstruction
# ------------------------------------------------------------
def _roundtrip_plane_tensor(
    plane: torch.Tensor,                    # [1,C,H,W] float32 (CPU)
    packing_mode: str,
    quant_mode: str,
    global_range: Tuple[float, float],
    r: int, c: int,
    align: int,
    jpeg_quality: int,
    emit_dir: str = "",
    plane_stem: str = "",
    emit_png: bool = False,
    emit_jpeg: bool = False,
) -> Tuple[torch.Tensor, int, Tuple[int,int], Tuple[int,int]]:
    """
    Returns:
      rec_plane: [1,C,H,W] float32 (CPU)
      bits:      encoded bits from JPEG enc (padded canvas)
      Hpad,Wpad: padded canvas size actually encoded
      Hp,Wp:     unpadded canvas size used for unpack
    """
    assert plane.dim() == 4 and plane.shape[0] == 1, "expect [1,C,H,W]"
    _, C, H, W = plane.shape

    # 1) normalize to [0,1] as in training
    x01, cmin, scale = normalize_planes(plane, mode=quant_mode, global_range=tuple(global_range))

    # 2) pack planes → RGB canvas (float01)
    rgb01_pad, (Hp, Wp) = pack_planes_to_rgb(x01, align=align, mode=packing_mode, r=r, c=c)  # [1,3,Hpad,Wpad]
    Hpad, Wpad = int(rgb01_pad.shape[-2]), int(rgb01_pad.shape[-1])

    # 3) JPEG round-trip (mono vs color)
    is_mono = (packing_mode == "flatten")
    if is_mono:
        mono01 = rgb01_pad[0, 0].contiguous().cpu().numpy()    # Hpad x Wpad
        rec_mono01, bits = jpeg_roundtrip_mono(mono01, quality=int(jpeg_quality))
        rec_rgb01 = np.stack([rec_mono01, rec_mono01, rec_mono01], axis=-1)  # Hpad x Wpad x 3
    else:
        rgb01 = rgb01_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()     # Hpad x Wpad x 3 (RGB)
        bgr01 = rgb01[..., ::-1]                                             # -> BGR
        rec_bgr01, bits = jpeg_roundtrip_color(bgr01, quality=int(jpeg_quality))
        rec_rgb01 = rec_bgr01[..., ::-1]                                     # BGR->RGB

    # (optional) visualization dumps (decoded images)
    if emit_dir and (emit_png or emit_jpeg):
        _mkdir(emit_dir)
        if is_mono:
            if emit_png:
                _write_gray(os.path.join(emit_dir, f"{plane_stem}_rec.png"), rec_rgb01[..., 0])
            if emit_jpeg:
                cv2.imwrite(os.path.join(emit_dir, f"{plane_stem}_rec.jpg"),
                            _to_u8_from_f01(rec_rgb01[..., ::-1]),
                            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        else:
            if emit_png:
                _write_color(os.path.join(emit_dir, f"{plane_stem}_rec.png"), rec_rgb01)
            if emit_jpeg:
                cv2.imwrite(os.path.join(emit_dir, f"{plane_stem}_rec.jpg"),
                            _to_u8_from_f01(rec_rgb01[..., ::-1]),
                            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])

    # 4) Crop back to (Hp,Wp) *before* unpacking, then unpack → [1,C,H,W] in [0,1]
    y_crop = torch.from_numpy(rec_rgb01).permute(2, 0, 1)[None, ...]    # [1,3,Hpad,Wpad]
    y_crop = y_crop[..., :Hp, :Wp].contiguous()
    rec01 = unpack_rgb_to_planes(y_crop, C=C, orig_size=(Hp, Wp), mode=packing_mode, r=r, c=c)

    # 5) De-normalize back to feature range
    rec_plane = (rec01 * scale + cmin).to(torch.float32)

    # 6) Final sanity crop to original [H,W]
    if rec_plane.shape[-2:] != (H, W):
        rec_plane = rec_plane[..., :H, :W]

    return rec_plane, int(bits), (Hpad, Wpad), (Hp, Wp)


# ------------------------------------------------------------
# Group processing
# ------------------------------------------------------------
def _process_group(
    sd: Dict[str, torch.Tensor],
    keys: List[str],
    cfg: dict,
    out_imgs_dir: str,
    jpeg_quality: int,
    emit_png: bool,
    emit_jpeg: bool,
    group_name: str,
):
    """
    Returns:
      rec_dict: { plane_key: Tensor[1,C,H,W] }
      stats:    list of dicts with bits & canvas sizes per plane
    """
    rec_dict, stats = {}, []
    for k in keys:
        x = sd[k].detach().clone().cpu()  # [1,C,H,W]
        _, C, H, W = x.shape

        _validate_channels(C, cfg["packing_mode"], cfg["r"], cfg["c"], k)

        rec, bits, (Hpad, Wpad), (Hp, Wp) = _roundtrip_plane_tensor(
            plane=x,
            packing_mode=cfg["packing_mode"],
            quant_mode=cfg["quant_mode"],
            global_range=tuple(cfg["global_range"]),
            r=int(cfg["r"]), c=int(cfg["c"]),
            align=int(cfg["align"]),
            jpeg_quality=int(jpeg_quality),
            emit_dir=out_imgs_dir,
            plane_stem=k.replace('.', '_'),
            emit_png=emit_png,
            emit_jpeg=emit_jpeg,
        )
        rec_dict[k] = rec
        stats.append({
            "plane": k,
            "shape": [int(d) for d in x.shape],
            "encoded_canvas_hw": [int(Hpad), int(Wpad)],
            "unpadded_hw_for_unpack": [int(Hp), int(Wp)],
            "bits": int(bits),
            "bpp": float(bits) / float(Hp * Wp) if Hp > 0 and Wp > 0 else 0.0,
        })
    return rec_dict, stats

# ------------------------------------------------------------
# CLI & main
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='Path to a TensoRF checkpoint (.pth/.th/.tar)')
    ap.add_argument('--outdir', required=True, help='Where to write the reconstructed checkpoint + summaries')
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

    # output images (optional; decoded frames for inspection only)
    ap.add_argument('--emit_png',  action='store_true')
    ap.add_argument('--emit_jpeg', action='store_true')
    ap.add_argument('--jpeg_quality', type=int, default=80)

    return ap.parse_args()


def main():
    args = parse_args()
    _mkdir(args.outdir)

    # Load ckpt (supports either pure state_dict or container with 'state_dict')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict'].copy()
    else:
        sd = ckpt.copy()

    den_keys = _plane_keys(sd, 'density_plane.')
    app_keys = _plane_keys(sd, 'app_plane.')

    den_cfg = dict(
        packing_mode=args.den_packing_mode,
        quant_mode=args.den_quant_mode,
        global_range=(float(args.den_global_range[0]), float(args.den_global_range[1])),
        r=int(args.den_r), c=int(args.den_c),
        align=int(args.align),
    )
    app_cfg = dict(
        packing_mode=args.app_packing_mode,
        quant_mode=args.app_quant_mode,
        global_range=(float(args.app_global_range[0]), float(args.app_global_range[1])),
        r=int(args.app_r), c=int(args.app_c),
        align=int(args.align),
    )

    # Optional decoded image dumps for inspection
    dump_den_dir = os.path.join(args.outdir, 'decoded_images', 'density')
    dump_app_dir = os.path.join(args.outdir, 'decoded_images', 'appearance')

    # Process density & appearance groups
    den_rec, den_stats = _process_group(
        sd, den_keys, den_cfg, dump_den_dir,
        jpeg_quality=args.jpeg_quality,
        emit_png=args.emit_png, emit_jpeg=args.emit_jpeg,
        group_name="density",
    )
    app_rec, app_stats = _process_group(
        sd, app_keys, app_cfg, dump_app_dir,
        jpeg_quality=args.jpeg_quality,
        emit_png=args.emit_png, emit_jpeg=args.emit_jpeg,
        group_name="appearance",
    )

    # Merge back into state dict
    for k, v in {**den_rec, **app_rec}.items():
        sd[k] = v

    # Write reconstructed checkpoint to outdir
    in_stem = Path(args.ckpt).name
    # preserve extension; add suffix
    suffix = f"_recon_jpeg{int(args.jpeg_quality)}.pth"
    if in_stem.endswith(('.pth', '.pt', '.tar', '.th')):
        base = Path(in_stem).stem
        out_ckpt = os.path.join(args.outdir, base + suffix)
    else:
        out_ckpt = os.path.join(args.outdir, in_stem + suffix)

    if 'state_dict' in ckpt:
        ckpt['state_dict'] = sd
        torch.save(ckpt, out_ckpt)
    else:
        torch.save(sd, out_ckpt)

    # Summaries: bits & simple text file
    def _sum_bits(stats_list):
        total_bits = sum(int(x["bits"]) for x in stats_list)
        total_MB   = total_bits / 8.0 / 1e6
        return total_bits, total_MB

    den_total_bits, den_total_MB = _sum_bits(den_stats)
    app_total_bits, app_total_MB = _sum_bits(app_stats)
    grand_bits = den_total_bits + app_total_bits
    grand_MB   = den_total_bits/8/1e6 + app_total_bits/8/1e6

    summary = {
        "jpeg_quality": int(args.jpeg_quality),
        "density": {
            "planes": den_stats,
            "total_bits": int(den_total_bits),
            "total_MB": float(den_total_MB),
        },
        "appearance": {
            "planes": app_stats,
            "total_bits": int(app_total_bits),
            "total_MB": float(app_total_MB),
        },
        "total_bits_all": int(grand_bits),
        "total_MB_all": float(grand_MB),
    }
    with open(os.path.join(args.outdir, "jpeg_bits_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.outdir, "bitrate.txt"), "w") as f:
        f.write(f"JPEG quality: {int(args.jpeg_quality)}\n")
        f.write(f"Density  : {den_total_bits} bits ({den_total_MB:.6f} MB)\n")
        f.write(f"Appearance: {app_total_bits} bits ({app_total_MB:.6f} MB)\n")
        f.write(f"TOTAL    : {grand_bits} bits ({grand_MB:.6f} MB)\n")

    print(f"[DONE] Reconstructed checkpoint written to: {out_ckpt}")
    print(f"[INFO] Bit summaries written to: {args.outdir}/bitrate.txt and jpeg_bits_summary.json")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
