#!/usr/bin/env python3
"""
python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_nerf_chair_jpeg_qp35

"""

import os, json, glob, pathlib, time, sys, argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import cv2

# ---- config / renderer / dataset / model imports (same ecosystem as eval_ste.py) ----
from opt import config_parser
from renderer import evaluation, OctreeRender_trilinear_fast
from utils import *
from dataLoader import dataset_dict
from models.tensorSTE import TensorSTE, PlanesCfg
from pathlib import Path

# ---- volumetric JPEG helpers (your recon_utils) ----
from models.recon_utils import (
    DCVC_ALIGN,
    normalize_planes,
    pack_planes_to_rgb,
    unpack_rgb_to_planes,
    jpeg_roundtrip_mono,
    jpeg_roundtrip_color,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_base_renderer = OctreeRender_trilinear_fast

# ------------------------------------------------------------
# Extra CLI parsing (NEW): --ckpt_dir and --train_iters
# ------------------------------------------------------------
def _parse_front_args():
    """
    Pre-parse a small set of extra CLI args that are NOT known to opt.config_parser,
    then strip them from sys.argv so config_parser() doesn't see them.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help=(
            "Directory containing ste_*_compression_<iters>.th and "
            "ste_*_system_<iters>.th (e.g. log3/ours_nerf_chair_jpeg_qp35)."
        ),
    )
    ap.add_argument(
        "--train_iters",
        type=int,
        default=29999,
        help=(
            "Training iteration index used in checkpoint filenames, "
            "e.g. 29999 for ste_*_compression_29999.th."
        ),
    )
    front_args, remaining = ap.parse_known_args(sys.argv[1:])
    # Remove these custom flags before config_parser() runs.
    sys.argv = [sys.argv[0]] + remaining
    return front_args


# ------------------------------------------------------------
# Small helpers (from your 3-in-1 script)
# ------------------------------------------------------------
def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _plane_keys(sd: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [k for k in sd.keys() if k.startswith(prefix)]
    def _idx(k):
        try:
            return int(k.split('.')[1])
        except Exception:
            return 0
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

def _to_u8_from_f01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)

def _write_gray(path: str, f01_hw: np.ndarray):
    ok = cv2.imwrite(path, _to_u8_from_f01(f01_hw))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")

def _write_color(path: str, f01_hw3_rgb: np.ndarray):
    # OpenCV expects BGR on write
    ok = cv2.imwrite(path, _to_u8_from_f01(f01_hw3_rgb[..., ::-1]))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


# ------------------------------------------------------------
# JPEG roundtrip for one plane tensor (from your 3-in-1 script)
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
# Group processing (from your 3-in-1)
# ------------------------------------------------------------
def _process_group(
    sd: Dict[str, torch.Tensor],
    keys: List[str],
    cfg: dict,
    jpeg_quality: int,
    emit_png: bool = False,
    emit_jpeg: bool = False,
    out_imgs_dir: str = "",
    group_name: str = "",
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
            plane_stem=f"{group_name}_{k.replace('.', '_')}",
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
# Some helpers from eval_ste.py (slightly trimmed)
# ------------------------------------------------------------
# ------------------------------------------------------------
# Some helpers from eval_ste.py (slightly trimmed)
# ------------------------------------------------------------
def _latest_by_mtime(pattern: str) -> str:
    paths = glob.glob(pattern)
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]

def _auto_find_ckpts(eval_root: str) -> Tuple[str, str]:
    """
    Find *_compression_*.th and *_system_*.th under eval_root (non-recursive).
    Returns (ckpt_model_only, ckpt_system).
    """
    comp = _latest_by_mtime(os.path.join(eval_root, "*_compression_*.th"))
    syst = _latest_by_mtime(os.path.join(eval_root, "*_system_*.th"))
    return comp or "", syst or ""

def _find_ckpts_from_dir(ckpt_dir: str, train_iters: int) -> Tuple[str, str]:
    """
    NEW: Find explicit *compression_<iters>.th and *system_<iters>.th
    under ckpt_dir, matching your current manual usage.

    NOTE: We preserve your existing mapping:
      system_ckpt <-- *compression_<iters>.th
      ckpt       <-- *system_<iters>.th
    """
    it_str = str(train_iters)
    comp_pattern = os.path.join(ckpt_dir, f"*compression_{it_str}.th")
    syst_pattern = os.path.join(ckpt_dir, f"*system_{it_str}.th")

    comp_path = _latest_by_mtime(comp_pattern)
    syst_path = _latest_by_mtime(syst_pattern)

    if not comp_path:
        raise FileNotFoundError(
            f"[4in1] No 'compression' checkpoint matching '{comp_pattern}' in {ckpt_dir}"
        )
    if not syst_path:
        raise FileNotFoundError(
            f"[4in1] No 'system' checkpoint matching '{syst_pattern}' in {ckpt_dir}"
        )

    return comp_path, syst_path

def _config_stem(args) -> str:
    cfg_path = getattr(args, "config", "") or ""
    if not cfg_path:
        return "eval"
    return pathlib.Path(cfg_path).stem

def _set_seed(seed: int = 20211202):
    import random, os as _os
    random.seed(seed)
    np.random.seed(seed)
    _os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _build_planescfg_from_args(args) -> PlanesCfg:
    return PlanesCfg(
        align=getattr(args, "align", DCVC_ALIGN),
        codec=str(getattr(args, "codec_backend", "jpeg")).lower(),
        vid_pix_fmt=str(getattr(args, "vid_pix_fmt", "yuv444p")),
        # density
        den_packing_mode=args.den_packing_mode,
        den_quant_mode=args.den_quant_mode,
        den_global_range=(args.den_global_min, args.den_global_max),
        den_r=args.den_r, den_c=args.den_c,
        den_quality=getattr(args, "den_quality", 80),
        den_png_level=getattr(args, "den_png_level", 6),
        den_hevc_qp=getattr(args, "den_hevc_qp", 32),
        den_hevc_preset=str(getattr(args, "den_hevc_preset", "medium")),
        den_av1_qp=getattr(args, "den_av1_qp", 36),
        den_av1_speed=getattr(args, "den_av1_speed", 6),
        den_vp9_qp=getattr(args, "den_vp9_qp", 40),
        den_vp9_speed=getattr(args, "den_vp9_speed", 4),
        # appearance
        app_packing_mode=args.app_packing_mode,
        app_quant_mode=args.app_quant_mode,
        app_global_range=(args.app_global_min, args.app_global_max),
        app_r=args.app_r, app_c=args.app_c,
        app_quality=getattr(args, "app_quality", 80),
        app_png_level=getattr(args, "app_png_level", 6),
        app_hevc_qp=getattr(args, "app_hevc_qp", 32),
        app_hevc_preset=str(getattr(args, "app_hevc_preset", "medium")),
        app_av1_qp=getattr(args, "app_av1_qp", 36),
        app_av1_speed=getattr(args, "app_av1_speed", 6),
        app_vp9_qp=getattr(args, "app_vp9_qp", 40),
        app_vp9_speed=getattr(args, "app_vp9_speed", 4),
    )


def _build_model_from_ckpt_ste_with_overrides(args, plane_overrides: Dict[str, torch.Tensor]):
    """
    Same spirit as _build_model_from_ckpt_ste in eval_ste.py, but:
      - patches the checkpoint's state_dict with JPEG-reconstructed planes
        before loading into the model.
      - disables external codec path (we already compressed offline).
    """
    system_path = getattr(args, "system_ckpt", "") or ""
    model_path  = getattr(args, "ckpt", "") or ""

    # If anything missing, auto-detect (same as eval_ste).
    eval_root = getattr(args, "eval_dir", "") or os.path.join(args.basedir, args.expname)
    if (not system_path) or (not model_path):
        auto_model, auto_system = _auto_find_ckpts(eval_root)
        if not model_path:
            model_path = auto_model
        if not system_path:
            system_path = auto_system
        setattr(args, "ckpt", model_path)
        setattr(args, "system_ckpt", system_path)

    if not system_path and not model_path:
        raise FileNotFoundError(
            f"No checkpoints found in '{eval_root}'. "
            f"Expected *_compression_*.th and/or *_system_*.th."
        )

    if system_path and os.path.exists(system_path):
        # Prefer system checkpoint (contains kwargs, alphaMask, etc.)
        ckpt = torch.load(system_path, map_location=device, weights_only=False)
        print(f"[4in1] using system checkpoint for render: {system_path}")
    else:
        # Fallback to model-only checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        print(f"[4in1] using model-only checkpoint for render: {model_path}")

    if "kwargs" not in ckpt:
        raise KeyError("Checkpoint must contain 'kwargs' to build TensorSTE; got keys: "
                       f"{list(ckpt.keys())[:10]} ...")

    # Patch planes in the checkpoint's state_dict BEFORE loading into model
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        for k, v in plane_overrides.items():
            if k in sd:
                sd[k] = v.to(device)
    else:
        sd = ckpt
        for k, v in plane_overrides.items():
            if k in sd:
                sd[k] = v.to(device)

    # Build model same way as eval_ste
    kwargs = dict(ckpt["kwargs"])
    kwargs.update({"device": device})
    model = TensorSTE(**kwargs).to(device)

    # STE plumbing (even if we're not using external codec anymore, this keeps layout identical)
    planescfg = _build_planescfg_from_args(args)
    model.init_ste(planescfg)
    model.set_ste(True)
    if hasattr(model, "enable_vec_qat"):
        model.enable_vec_qat()

    # IMPORTANT: disable internal external codec path; we already did JPEG offline
    model.using_external_codec = False
    if hasattr(model, "set_compress_before_volrend"):
        model.set_compress_before_volrend(False)
    else:
        model.compress_before_volrend = False

    # Finally load patched ckpt
    model.load(ckpt)
    return model


# ------------------------------------------------------------
# Main 4-in-1 pipeline
# ------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float32)
    front = _parse_front_args()
    args = config_parser()
    args.ckpt_dir    = front.ckpt_dir
    args.train_iters = front.train_iters

    # Seed / determinism
    seed = getattr(args, "seed", 20211202)
    _set_seed(seed)

    # -------------------- eval dir layout (same spirit as eval_ste) --------------------
    # print(args.ckpt_dir)
    # raise Exception
    if args.ckpt_dir is not None:
        # We want to replicate your current manual usage:
        #   --system_ckpt  ste_..._compression_<iters>.th
        #   --ckpt         ste_..._system_<iters>.th
        comp_ckpt, syst_ckpt = _find_ckpts_from_dir(args.ckpt_dir, args.train_iters)
        args.system_ckpt = comp_ckpt   # compression_* -> system_ckpt (same as your example)
        args.ckpt        = syst_ckpt   # system_*      -> ckpt
        eval_root = args.ckpt_dir
        # Also propagate as eval_dir for any fallback code
        args.eval_dir = eval_root

        print("[4in1] auto-resolved checkpoints from ckpt_dir:")
        print(f"       ckpt_dir    : {args.ckpt_dir}")
        print(f"       system_ckpt : {args.system_ckpt}")
        print(f"       ckpt        : {args.ckpt}")
    else:
        eval_root = str(Path(args.system_ckpt).parent)

    cfg_stem  = _config_stem(args)
    eval_dir  = os.path.join(eval_root, "eval_jpeg")
    _mkdir(eval_dir)
    print(f"[4in1] eval_dir: {eval_dir}")

    # Save full config for reproducibility
    with open(os.path.join(eval_dir, "eval_cfg.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------- JPEG on volumetric planes (3-in-1 logic, in-memory) --------------------
    # We compress from the model-only ckpt if given; else system ckpt; else auto-detect.
    codec_ckpt_path = getattr(args, "ckpt", "") or ""
    if not codec_ckpt_path:
        auto_model, auto_system = _auto_find_ckpts(eval_root)
        codec_ckpt_path = auto_model or auto_system
        if not codec_ckpt_path:
            raise FileNotFoundError(
                f"[4in1] No checkpoint found in '{eval_root}' for codec planes."
            )
        setattr(args, "ckpt", codec_ckpt_path)

    print(f"[4in1] codec plane source: {codec_ckpt_path}")
    ckpt_codec = torch.load(codec_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_codec, dict) and "state_dict" in ckpt_codec:
        sd = ckpt_codec["state_dict"].copy()
    else:
        sd = ckpt_codec.copy()

    den_keys = _plane_keys(sd, "density_plane.")
    app_keys = _plane_keys(sd, "app_plane.")

    # Build configs from training args
    align_val = getattr(args, "align", DCVC_ALIGN)

    den_cfg = dict(
        packing_mode=args.den_packing_mode,
        quant_mode=args.den_quant_mode,
        global_range=(float(args.den_global_min), float(args.den_global_max)),
        r=int(args.den_r), c=int(args.den_c),
        align=int(align_val),
    )
    app_cfg = dict(
        packing_mode=args.app_packing_mode,
        quant_mode=args.app_quant_mode,
        global_range=(float(args.app_global_min), float(args.app_global_max)),
        r=int(args.app_r), c=int(args.app_c),
        align=int(align_val),
    )

    # No intermediate plane PNGs/JPEGs by default
    emit_den_dir = ""
    emit_app_dir = ""
    emit_png  = False
    emit_jpeg = False

    print("[4in1] JPEG roundtrip for density planes …")
    den_rec, den_stats = _process_group(
        sd, den_keys, den_cfg,
        jpeg_quality=int(getattr(args, "den_quality", 80)),
        emit_png=emit_png, emit_jpeg=emit_jpeg,
        out_imgs_dir=emit_den_dir,
        group_name="density",
    )
    print("[4in1] JPEG roundtrip for appearance planes …")
    app_rec, app_stats = _process_group(
        sd, app_keys, app_cfg,
        jpeg_quality=int(getattr(args, "app_quality", 80)),
        emit_png=emit_png, emit_jpeg=emit_jpeg,
        out_imgs_dir=emit_app_dir,
        group_name="appearance",
    )

    # Merge rec planes (used later to patch model ckpt)
    plane_overrides = {**den_rec, **app_rec}

    # ---- bitrate summaries (matching your 3-in-1 style) ----
    def _sum_bits(stats_list):
        total_bits = sum(int(x["bits"]) for x in stats_list)
        total_MB   = total_bits / 8.0 / 1e6
        return total_bits, total_MB

    den_total_bits, den_total_MB = _sum_bits(den_stats)
    app_total_bits, app_total_MB = _sum_bits(app_stats)
    grand_bits = den_total_bits + app_total_bits
    grand_MB   = grand_bits / 8.0 / 1e6

    summary = {
        "den_quality": int(getattr(args, "den_quality", 80)),
        "app_quality": int(getattr(args, "app_quality", 80)),
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
    with open(os.path.join(eval_dir, "jpeg_bits_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(eval_dir, "bitrate.txt"), "w") as f:
        f.write(f"Density  : {den_total_bits} bits ({den_total_MB:.6f} MB)\n")
        f.write(f"Appearance: {app_total_bits} bits ({app_total_MB:.6f} MB)\n")
        f.write(f"TOTAL    : {grand_bits} bits ({grand_MB:.6f} MB)\n")

    print(f"[4in1] bitrate summary written to: {eval_dir}/bitrate.txt")

    # -------------------- dataset (same as eval_ste) --------------------
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split="test",
                           downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray  = args.ndc_ray

    # -------------------- model with JPEG-reconstructed planes --------------------
    tensorf = _build_model_from_ckpt_ste_with_overrides(args, plane_overrides)
    tensorf.to(device)

    # nSamples policy identical to eval_ste
    if getattr(args, "compression", True):
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, getattr(tensorf, "nSamples", args.nSamples))

    # -------------------- render test views & PSNR --------------------
    save_img_dir = os.path.join(eval_dir, "imgs_test")
    _mkdir(save_img_dir)
    n_vis = args.N_vis if getattr(args, "N_vis", 0) and args.N_vis > 0 else -1

    with torch.no_grad():
        PSNRs = evaluation(
            test_dataset, tensorf, args, _base_renderer, save_img_dir,
            N_vis=n_vis,
            prtx="jpeg_",
            N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray,
            compute_extra_metrics=False,
            device=device,
        )

    avg_psnr = float(np.mean(PSNRs)) if len(PSNRs) else float("nan")
    with open(os.path.join(eval_dir, "average_psnr.txt"), "w") as f:
        f.write(f"{avg_psnr:.6f}\n")
    print(f"[4in1] average PSNR over {len(PSNRs)} views: {avg_psnr:.4f} dB")
    print(f"[4in1] DONE. Outputs in: {eval_dir}")


if __name__ == "__main__":
    main()
