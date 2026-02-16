"""
Usage:
python eval_ste.py  \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg35.txt \
    --system_ckpt log_2/nerf_chair_jpeg35/tensorste_chair_jpeg35_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg35/tensorste_chair_jpeg35_system_19999.th 

python eval_ste.py  \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg65.txt \
    --system_ckpt log_2/nerf_chair_jpeg65/tensorste_chair_jpeg65_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg65/tensorste_chair_jpeg65_system_19999.th 

python eval_ste.py  \
    --dataset_name blender \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg80.txt \
    --system_ckpt log_2/nerf_chair_jpeg80/tensorste_chair_jpeg80_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg80/tensorste_chair_jpeg80_system_19999.th \
    --eval_dir qualitative/nerf_chair_jpeg80

    python eval_ste.py  \
    --dataset_name blender \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg65.txt \
    --system_ckpt log_2/nerf_chair_jpeg65/tensorste_chair_jpeg65_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg65/tensorste_chair_jpeg65_system_19999.th \
    --eval_dir qualitative/nerf_chair_jpeg65

python eval_ste.py  \
    --dataset_name blender \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg50.txt \
    --system_ckpt log_2/nerf_chair_jpeg50/tensorste_chair_jpeg50_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg50/tensorste_chair_jpeg50_system_19999.th \
    --eval_dir qualitative/nerf_chair_jpeg50

python eval_ste.py  \
    --dataset_name blender \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg35.txt \
    --system_ckpt log_2/nerf_chair_jpeg35/tensorste_chair_jpeg35_compression_19999.th \
    --ckpt log_2/nerf_chair_jpeg35/tensorste_chair_jpeg35_system_19999.th \
    --eval_dir qualitative/nerf_chair_jpeg35

python eval_ste.py  \
    --dataset_name blender \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair_codec_ste_jpeg20.txt \
    --system_ckpt log3/ours_nerf_chair_jpeg_qp20/te_nerf_chair_jpeg_qp20_compression_29999.th \
    --ckpt log3/ours_nerf_chair_jpeg_qp20/ste_nerf_chair_jpeg_qp20_system_29999.th \
    --eval_dir qualitative/nerf_chair_jpeg20
"""
#!/usr/bin/env python3
import os, json, glob
import pathlib, time
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Tuple

# === your codebase imports (same as training) ===
from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict
from models.tensorSTE import TensorSTE, PlanesCfg

# --------------------------------------------------------------------------------------
# Helpers copied/adapted from your STE training script so eval matches training
# --------------------------------------------------------------------------------------
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_base_renderer = OctreeRender_trilinear_fast  # unchanged core

def _latest_by_mtime(pattern: str) -> str:
    """Return the latest file (by mtime) matching glob pattern, else ''."""
    paths = glob.glob(pattern)
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]

def _auto_find_ckpts(eval_root: str) -> Tuple[str, str]:
    """
    Find *_compression_*.th and *_system_*.th under eval_root (non-recursive).
    Returns (ckpt_model_only, ckpt_system). Empty string if missing.
    """
    comp = _latest_by_mtime(os.path.join(eval_root, "*_compression_*.th"))
    syst = _latest_by_mtime(os.path.join(eval_root, "*_system_*.th"))
    # In your training naming, *_compression_*.th is the model-only, *_system_*.th is the system ckpt.
    return comp or "", syst or ""

def _config_stem(args) -> str:
    """
    Best-effort to get the base name (without extension) of the provided --config file.
    Falls back to 'eval' if unavailable.
    """
    cfg_path = getattr(args, "config", "") or ""
    if not cfg_path:
        return "eval"
    return pathlib.Path(cfg_path).stem

def set_seed(seed: int = 20211202):
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
        align=getattr(args, "align", 32),
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

def _maybe_guess_system_ckpt(model_ckpt_path: str) -> str:
    if not model_ckpt_path:
        return ""
    root = os.path.dirname(model_ckpt_path)
    base = os.path.basename(model_ckpt_path)
    exp  = base.split("_compression_")[0]
    cand = sorted(glob.glob(os.path.join(root, f"{exp}_system_*.th")))
    return cand[-1] if cand else ""

def _build_model_from_ckpt_ste(args):
    """Prefer system ckpt; else load model-only and init STE, exactly like training."""
    system_path = getattr(args, "system_ckpt", "") or ""
    if (not system_path) and getattr(args, "ckpt", ""):
        guess = _maybe_guess_system_ckpt(args.ckpt)
        if guess:
            print(f"[eval] Using inferred system checkpoint: {guess}")
            system_path = guess

    if system_path and os.path.exists(system_path):
        print(f"[eval] loading system checkpoint: {system_path}")
        system = torch.load(system_path, map_location=device, weights_only=False)
        kwargs = dict(system["kwargs"]); kwargs.update({"device": device})
        model  = TensorSTE(**kwargs).to(device)
        cfg    = _build_planescfg_from_args(args)
        model.init_ste(cfg)
        model.set_ste(True)
        if hasattr(model, "enable_vec_qat"):
            model.enable_vec_qat()
        model.load(system)  # restores alphaMask etc.
        return model

    if not getattr(args, "ckpt", ""):
        raise FileNotFoundError("Neither --system_ckpt nor --ckpt provided.")

    print(f"[eval] loading model-only checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = dict(ckpt["kwargs"]); kwargs.update({"device": device})
    model  = TensorSTE(**kwargs).to(device)
    cfg    = _build_planescfg_from_args(args)
    model.init_ste(cfg)
    model.set_ste(True)
    if hasattr(model, "enable_vec_qat"):
        model.enable_vec_qat()
    model.load(ckpt)
    return model

# ---- Renderer wrapper: ensure codec reconstruction planes exist on every render call ----
def STE_safe_renderer(rays, tensorf, *args, **kwargs):
    # Ensure we are in the external-codec path and populate recon planes
    if getattr(tensorf, "compress_before_volrend", False) and getattr(tensorf, "using_external_codec", False):
        with torch.no_grad():
            tensorf.compress_with_external_codec(mode="eval")
    return _base_renderer(rays, tensorf, *args, **kwargs)

# --------------------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)

    # Parse the same config you used for training (so all codec params are present)
    args = config_parser()
    print(args)

    # -------------------- resolve eval root + config-named subdir --------------------
    # eval_root is what you pass via --eval_dir (e.g., log_2/nerf_chair)
    eval_root = getattr(args, "eval_dir", "") or os.path.join(args.basedir, args.expname)
    cfg_stem  = _config_stem(args)  # e.g., "chair_codec_ste_jpeg35"
    eval_dir  = os.path.join(eval_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Persist full cfg for reproducibility
    with open(os.path.join(eval_dir, "eval_cfg.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------- auto-detect checkpoints if not provided --------------------
    sys_ckpt = getattr(args, "system_ckpt", "") or ""
    mdl_ckpt = getattr(args, "ckpt", "") or ""

    if (not sys_ckpt) or (not mdl_ckpt):
        auto_model, auto_system = _auto_find_ckpts(eval_root)
        if not mdl_ckpt:
            mdl_ckpt = auto_model
        if not sys_ckpt:
            sys_ckpt = auto_system
        # Write back into args so downstream code sees them
        setattr(args, "ckpt", mdl_ckpt)
        setattr(args, "system_ckpt", sys_ckpt)

        print(f"[eval:auto] ckpt(model-only *_compression_*.th): {mdl_ckpt or '(not found)'}")
        print(f"[eval:auto] system_ckpt(*_system_*.th):         {sys_ckpt or '(not found)'}")

        # If neither found, fail early with a clear message
        if not sys_ckpt and not mdl_ckpt:
            raise FileNotFoundError(
                f"No checkpoints found in '{eval_root}'. "
                f"Expected *_compression_*.th and/or *_system_*.th."
            )


    # -------------------- dataset (match training) --------------------
    dataset = dataset_dict[args.dataset_name]
    test_dataset  = dataset(args.datadir, split="test", downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray  = args.ndc_ray

    # -------------------- model (match training) --------------------
    tensorf = _build_model_from_ckpt_ste(args)
    # harden the flags to match training’s render path
    tensorf.using_external_codec = True
    if hasattr(tensorf, "set_compress_before_volrend"):
        tensorf.set_compress_before_volrend(True)
    else:
        tensorf.compress_before_volrend = True
    if hasattr(tensorf, "set_codec_cache"):
        tensorf.set_codec_cache(
            refresh_k=getattr(args, "refresh_k", 1),
            refresh_eps=getattr(args, "refresh_eps", 0.0),
            bpp_refresh_k=getattr(args, "refresh_k", 1),
        )

    # nSamples policy identical to training
    if getattr(args, "compression", True):
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, getattr(tensorf, "nSamples", args.nSamples))

    # -------------------- one-time codec stats & bitrate --------------------
    den_bits = app_bits = 0
    if getattr(args, "compression", True):
        with torch.no_grad():
            out = tensorf.compress_with_external_codec(mode="eval")
        den_bits = sum(int(p["bits"]) for p in out["den"]["rec_likelihood"])
        app_bits = sum(int(p["bits"]) for p in out["app"]["rec_likelihood"])
        total_bits = den_bits + app_bits
        with open(os.path.join(eval_dir, "bitrate.txt"), "w") as f:
            f.write(f"den_bits:   {den_bits}\n")
            f.write(f"app_bits:   {app_bits}\n")
            f.write(f"total_bits: {total_bits}\n")
        print(f"[eval] bits -> den: {den_bits}  app: {app_bits}  total: {total_bits}")

    # -------------------- render N_vis test views (same evaluation() as training) --------------------
    save_img_dir = os.path.join(eval_dir, "imgs_test")
    os.makedirs(save_img_dir, exist_ok=True)
    n_vis = args.N_vis if getattr(args, "N_vis", 0) and args.N_vis > 0 else -1

    with torch.no_grad():
        PSNRs = evaluation(
            test_dataset, tensorf, args, STE_safe_renderer, save_img_dir,
            N_vis=n_vis,
            prtx=f"{getattr(args, 'codec_backend', 'ste')}_",
            N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray,
            compute_extra_metrics=False,
            device=device,
        )

    avg_psnr = float(np.mean(PSNRs)) if len(PSNRs) else float("nan")
    with open(os.path.join(eval_dir, "average_psnr.txt"), "w") as f:
        f.write(f"{avg_psnr:.6f}\n")
    print(f"[eval] average PSNR over {len(PSNRs)} views: {avg_psnr:.4f} dB")

if __name__ == "__main__":
    main()
