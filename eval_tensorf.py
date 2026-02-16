#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
from typing import Tuple

import numpy as np
import torch

# ==== imports from your codebase ====
from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast  # keep identical to training

"""
Usage:
python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair.txt \
    --ckpt_dir log/tensorf_chair_VM

"""
# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def set_seed(seed: int = 20211202):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


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
            "Folder containing the vanilla TensoRF checkpoint, e.g. "
            "log/tensorf_chair_VM. The script will search for '*_VM.th' inside."
        ),
    )
    ap.add_argument(
        "--ckpt_suffix",
        type=str,
        default="_VM.th",
        help="Filename suffix used to locate the baseline checkpoint (default: _VM.th).",
    )
    front_args, remaining = ap.parse_known_args(sys.argv[1:])
    # Remove these flags before config_parser() runs
    sys.argv = [sys.argv[0]] + remaining
    return front_args


def _find_tensorf_ckpt(ckpt_dir: str, suffix: str) -> str:
    """
    Find the vanilla TensoRF checkpoint under ckpt_dir whose filename ends with suffix,
    e.g. '*_VM.th'. If multiple match, choose the latest by mtime.
    """
    pattern = os.path.join(ckpt_dir, f"*{suffix}")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"[baseline] No checkpoint found in '{ckpt_dir}' matching pattern '{pattern}'"
        )
    candidates.sort(key=lambda p: os.path.getmtime(p))
    ckpt_path = candidates[-1]
    print(f"[baseline] Using checkpoint: {ckpt_path}")
    return ckpt_path


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)

    # First, parse our extra arguments (ckpt_dir / ckpt_suffix)
    front = _parse_front_args()

    # Then parse the standard training/eval config
    args = config_parser()

    # Attach our custom args so they appear in eval_cfg.json
    args.ckpt_dir = front.ckpt_dir
    args.ckpt_suffix = front.ckpt_suffix

    # ---- Resolve checkpoint path ----
    if args.ckpt_dir is not None:
        ckpt_dir = os.path.abspath(args.ckpt_dir)
        ckpt_path = _find_tensorf_ckpt(ckpt_dir, args.ckpt_suffix)
        args.ckpt = ckpt_path
        eval_root = ckpt_dir
    else:
        # Fallback: use args.ckpt directly (must be provided via CLI/config)
        if not getattr(args, "ckpt", None):
            raise FileNotFoundError(
                "[baseline] Neither --ckpt_dir nor --ckpt provided. "
                "Please pass --ckpt_dir log/tensorf_chair_VM or --ckpt path/to/ckpt.th"
            )
        ckpt_path = args.ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[baseline] ckpt not found: {ckpt_path}")
        eval_root = os.path.dirname(os.path.abspath(ckpt_path))

    # ---- Eval directory ----
    eval_dir = os.path.join(eval_root, "eval_tensorf")
    _mkdir(eval_dir)
    print(f"[baseline] eval_dir: {eval_dir}")

    # Save full config for reproducibility
    with open(os.path.join(eval_dir, "eval_cfg.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------- dataset (same as training) --------------------
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    # -------------------- load vanilla TensoRF model --------------------
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"[baseline] ckpt path does not exist: {args.ckpt}")

    print(f"[baseline] Loading TensoRF checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})

    # (For vanilla TensoRF baseline, compression flags should be False/unused.)
    if getattr(args, "rate_penalty", False):
        kwargs.update({"rate_penalty": True})
    if getattr(args, "compression", False):
        kwargs.update({"compression_strategy": args.compression_strategy})

    tensorf = eval(args.model_name)(**kwargs).to(device)
    tensorf.load(ckpt)

    # If the baseline was trained with compression enabled, we keep original behavior.
    if getattr(args, "compression", False):
        if args.compression_strategy == "batchwise_img_coding":
            tensorf.init_image_codec()
        elif args.compression_strategy == "adaptor_feat_coding":
            tensorf.init_feat_codec()

    # -------------------- render N_vis test views & PSNR --------------------
    save_img_dir = os.path.join(eval_dir, "imgs_test")
    _mkdir(save_img_dir)

    # Default to N_vis=5 if not set or <=0
    if getattr(args, "N_vis", 0) and args.N_vis > 0:
        n_vis = args.N_vis
    else:
        n_vis = 5

    print(f"[baseline] Evaluating {n_vis} test views…")

    with torch.no_grad():
        PSNRs = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            save_img_dir,
            N_vis=n_vis,
            prtx="tensorf_",
            N_samples=-1,       # let evaluation use its default policy
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            compute_extra_metrics=True,
            device=device,
        )

    avg_psnr = float(np.mean(PSNRs)) if len(PSNRs) else float("nan")
    with open(os.path.join(eval_dir, "average_psnr.txt"), "w") as f:
        f.write(f"{avg_psnr:.6f}\n")

    print(f"[baseline] average PSNR over {len(PSNRs)} views: {avg_psnr:.4f} dB")
    print(f"[baseline] DONE. Outputs in: {eval_dir}")


if __name__ == "__main__":
    main()
