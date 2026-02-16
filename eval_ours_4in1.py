#!/usr/bin/env python3
import os, json, glob, pathlib, sys, argparse
from typing import Tuple, List, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

# === your codebase imports (same as training / eval_ste.py) ===
from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict
from models.tensorSTE import TensorSTE, PlanesCfg

from pathlib import Path

# --------------------------------------------------------------------------------------
# Globals / base renderer
# --------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_base_renderer = OctreeRender_trilinear_fast  # unchanged core

"""
python eval_ours_4in1.py \
  --dataset_name tankstemple \
  --N_vis 5 \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family\
  --config configs/tt_family/family_codec_ste_jpeg65.txt \
  --ckpt_dir log3/ours_tt_family_jpeg_qp65 

python eval_ours_4in1.py \
  --dataset_name blender \
  --N_vis 8 \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/lego\
  --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
  --ckpt_dir log3/ours_nerf_lego_jpeg_qp65

python eval_ours_4in1.py \
  --dataset_name blender \
  --N_vis 8 \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials\
  --config configs/nerf_materials/materials_codec_ste_jpeg65.txt \
  --ckpt_dir log3/ours_nerf_materials_jpeg_qp65


python eval_ours_4in1.py \
  --dataset_name tankstemple \
  --N_vis 8 \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck\
  --config configs/tt_truck/truck_codec_ste_jpeg80.txt \
  --ckpt_dir log3/ours_tt_truck_jpeg_qp80 
"""

# --------------------------------------------------------------------------------------
# Front-end parser: --ckpt_dir and --train_iters
# --------------------------------------------------------------------------------------
def _parse_front_args():
    """
    Pre-parse extra CLI args that are NOT known to opt.config_parser,
    then strip them from sys.argv so config_parser() doesn't see them.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help=(
            "Directory containing ste_*_compression_<iters>.th and "
            "ste_*_system_<iters>.th (e.g. log_2/nerf_chair_jpeg65)."
        ),
    )
    ap.add_argument(
        "--train_iters",
        type=int,
        default=None,
        help=(
            "Training iteration index used in checkpoint filenames, "
            "e.g. 19999 for ste_*_compression_19999.th. "
            "If not given, will fall back to latest *_compression_*.th / *_system_*.th."
        ),
    )
    front_args, remaining = ap.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return front_args


# --------------------------------------------------------------------------------------
# Helpers (mostly copied from eval_ste.py)
# --------------------------------------------------------------------------------------
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


def _find_ckpts_from_dir(ckpt_dir: str, train_iters: int) -> Tuple[str, str]:
    """
    Find explicit *compression_<iters>.th and *system_<iters>.th under ckpt_dir.

    IMPORTANT: We preserve your existing manual mapping:
      system_ckpt <-- *compression_<iters>.th
      ckpt       <-- *system_<iters>.th

    so behavior matches your original eval_ste.py usage:
      --system_ckpt tensorste_..._compression_19999.th
      --ckpt       tensorste_..._system_19999.th
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
        den_r=args.den_r,
        den_c=args.den_c,
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
        app_r=args.app_r,
        app_c=args.app_c,
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
    exp = base.split("_compression_")[0]
    cand = sorted(glob.glob(os.path.join(root, f"{exp}_system_*.th")))
    return cand[-1] if cand else ""


def _build_model_from_ckpt_ste(args):
    """
    Prefer system ckpt; else load model-only and init STE, exactly like training.

    This is copied from eval_ste.py so behavior is identical.
    """
    system_path = getattr(args, "system_ckpt", "") or ""
    if (not system_path) and getattr(args, "ckpt", ""):
        guess = _maybe_guess_system_ckpt(args.ckpt)
        if guess:
            print(f"[eval] Using inferred system checkpoint: {guess}")
            system_path = guess

    if system_path and os.path.exists(system_path):
        print(f"[eval] loading system checkpoint: {system_path}")
        system = torch.load(system_path, map_location=device, weights_only=False)
        kwargs = dict(system["kwargs"])
        kwargs.update({"device": device})
        model = TensorSTE(**kwargs).to(device)
        cfg = _build_planescfg_from_args(args)
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
    kwargs = dict(ckpt["kwargs"])
    kwargs.update({"device": device})
    model = TensorSTE(**kwargs).to(device)
    cfg = _build_planescfg_from_args(args)
    model.init_ste(cfg)
    model.set_ste(True)
    if hasattr(model, "enable_vec_qat"):
        model.enable_vec_qat()
    model.load(ckpt)
    return model


# ---- Renderer wrapper: ensure codec reconstruction planes exist on every render call ----
def STE_safe_renderer(rays, tensorf, *args, **kwargs):
    # Ensure we are in the external-codec path and populate recon planes
    if getattr(tensorf, "compress_before_volrend", False) and getattr(
        tensorf, "using_external_codec", False
    ):
        with torch.no_grad():
            tensorf.compress_with_external_codec(mode="eval")
    return _base_renderer(rays, tensorf, *args, **kwargs)


# --------------------------------------------------------------------------------------
# Main evaluation (eval_ste behavior + 4-in-1 I/O)
# --------------------------------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float32)

    # 1) Front args for ckpt_dir / train_iters
    front = _parse_front_args()

    # 2) Full config (same as training / eval_ste.py)
    args = config_parser()
    # Attach front args so they're visible in eval_cfg.json
    args.ckpt_dir = front.ckpt_dir
    args.train_iters = front.train_iters

    print(args)

    # Seed / determinism
    seed = getattr(args, "seed", 20211202)
    set_seed(seed)

    # -------------------- resolve eval_root and checkpoints --------------------
    if args.ckpt_dir is not None:
        eval_root = args.ckpt_dir

        if args.train_iters is not None:
            # Use explicit iters to match your manual usage
            comp_ckpt, syst_ckpt = _find_ckpts_from_dir(args.ckpt_dir, args.train_iters)
        else:
            # Fall back to latest *_compression_*.th / *_system_*.th
            comp_ckpt, syst_ckpt = _auto_find_ckpts(args.ckpt_dir)
            if not comp_ckpt or not syst_ckpt:
                raise FileNotFoundError(
                    f"[4in1] Could not auto-find *_compression_*.th / *_system_*.th in {args.ckpt_dir}"
                )

        # Preserve your original mapping:
        #   system_ckpt <-- *compression_*.th
        #   ckpt       <-- *system_*.th
        args.system_ckpt = comp_ckpt
        args.ckpt = syst_ckpt

        print("[4in1] auto-resolved checkpoints from ckpt_dir:")
        print(f"       ckpt_dir    : {args.ckpt_dir}")
        print(f"       system_ckpt : {args.system_ckpt}")
        print(f"       ckpt        : {args.ckpt}")
    else:
        # No ckpt_dir: behave like eval_ste.py (eval_root from basedir/expname or args.eval_dir)
        eval_root = getattr(args, "eval_dir", "") or os.path.join(args.basedir, args.expname)
        args.eval_dir = eval_root

    # Use a different subdir name from eval_ste ("eval_jpeg" instead of "eval")
    cfg_stem = _config_stem(args)
    eval_dir = os.path.join(eval_root, "eval_jpeg")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"[4in1] eval_dir: {eval_dir}")

    # Persist full cfg for reproducibility
    with open(os.path.join(eval_dir, "eval_cfg.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------- dataset (match training / eval_ste.py) --------------------
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    # -------------------- model (identical to eval_ste.py) --------------------
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

    tensorf.to(device)

    # nSamples policy identical to training / eval_ste.py
    if getattr(args, "compression", True):
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, getattr(tensorf, "nSamples", args.nSamples))

    # -------------------- codec stats & bitrate (one-time) --------------------
    den_bits = app_bits = 0
    den_stats: List[Dict] = []
    app_stats: List[Dict] = []

    if getattr(args, "compression", True):
        with torch.no_grad():
            out = tensorf.compress_with_external_codec(mode="eval")

        # out["den"]["rec_likelihood"] and out["app"]["rec_likelihood"] are the
        # same structures you used in training; here we summarize bits per entry.
        den_recs = out.get("den", {}).get("rec_likelihood", [])
        app_recs = out.get("app", {}).get("rec_likelihood", [])

        for i, p in enumerate(den_recs):
            b = int(p.get("bits", 0))
            den_bits += b
            # We only know bits for sure; canvas sizes etc. are optional and
            # depend on your implementation. We keep the schema compatible.
            den_stats.append(
                {
                    "index": int(i),
                    "bits": b,
                }
            )

        for i, p in enumerate(app_recs):
            b = int(p.get("bits", 0))
            app_bits += b
            app_stats.append(
                {
                    "index": int(i),
                    "bits": b,
                }
            )

        total_bits = den_bits + app_bits

        # Convert to MB (bytes / 1e6)
        den_MB = den_bits / 8.0 / 1e6
        app_MB = app_bits / 8.0 / 1e6
        total_MB = total_bits / 8.0 / 1e6

        # bitrate.txt in the "new script" style
        with open(os.path.join(eval_dir, "bitrate.txt"), "w") as f:
            f.write(f"Density  : {den_bits} bits ({den_MB:.6f} MB)\n")
            f.write(f"Appearance: {app_bits} bits ({app_MB:.6f} MB)\n")
            f.write(f"TOTAL    : {total_bits} bits ({total_MB:.6f} MB)\n")

        print(
            f"[4in1] bits -> den: {den_bits}  app: {app_bits}  total: {total_bits}"
        )

        # jpeg_bits_summary.json (structured summary, similar to your 4-in-1 script)
        summary = {
            "den_quality": int(getattr(args, "den_quality", 80)),
            "app_quality": int(getattr(args, "app_quality", 80)),
            "density": {
                "planes": den_stats,
                "total_bits": int(den_bits),
                "total_MB": float(den_MB),
            },
            "appearance": {
                "planes": app_stats,
                "total_bits": int(app_bits),
                "total_MB": float(app_MB),
            },
            "total_bits_all": int(total_bits),
            "total_MB_all": float(total_MB),
        }
        with open(os.path.join(eval_dir, "jpeg_bits_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # -------------------- render N_vis test views (same evaluation() as training) --------------------
    save_img_dir = os.path.join(eval_dir, "imgs_test")
    os.makedirs(save_img_dir, exist_ok=True)
    n_vis = args.N_vis if getattr(args, "N_vis", 0) and args.N_vis > 0 else -1

    with torch.no_grad():
        PSNRs = evaluation(
            test_dataset,
            tensorf,
            args,
            STE_safe_renderer,
            save_img_dir,
            N_vis=n_vis,
            prtx=f"{getattr(args, 'codec_backend', 'ste')}_",
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            compute_extra_metrics=True,
            device=device,
        )
        # PSNRs = evaluation_ds(
        #     test_dataset,
        #     tensorf,
        #     args,
        #     STE_safe_renderer,
        #     save_img_dir,
        #     N_vis=n_vis,
        #     prtx=f"{getattr(args, 'codec_backend', 'ste')}_",
        #     N_samples=nSamples,
        #     white_bg=white_bg,
        #     ndc_ray=ndc_ray,
        #     compute_extra_metrics=False,
        #     device=device,
        #     downsample=1.0
        # )

    avg_psnr = float(np.mean(PSNRs)) if len(PSNRs) else float("nan")
    with open(os.path.join(eval_dir, "average_psnr.txt"), "w") as f:
        f.write(f"{avg_psnr:.6f}\n")

    print(f"[4in1] average PSNR over {len(PSNRs)} views: {avg_psnr:.4f} dB")
    print(f"[4in1] DONE. Outputs in: {eval_dir}")


if __name__ == "__main__":
    main()
