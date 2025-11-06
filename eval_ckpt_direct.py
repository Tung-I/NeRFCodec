#!/usr/bin/env python3
import os, sys, json, math, random, datetime, glob
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path

# repo-local imports
from renderer import *
from utils import *
from dataLoader import dataset_dict

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast  # keep identical to training

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def set_seed(seed: int = 20211202):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _build_log_dir(args) -> str:
    if args.add_timestamp:
        return f"{args.basedir}/{args.expname}/{datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')}"
    if args.add_exp_version:
        root = f"{args.basedir}/{args.expname}"
        os.makedirs(root, exist_ok=True)
        versions = sorted(glob.glob(f"{root}/version_*"))
        idx = 0 if not versions else (int(versions[-1].split("_")[-1]) + 1)
        return f"{root}/version_{idx:03d}"
    return f"{args.basedir}/{args.expname}"

def _force_no_compression_kwargs(ckpt_kwargs: dict, args) -> dict:
    """
    We want a pure render of the stored planes; ensure no compression path is taken,
    even if the ckpt was produced by a codec-trained run.
    """
    kw = dict(ckpt_kwargs)
    kw["device"] = device
    # turn off any pre-render compression paths
    kw["compress_before_volrend"] = False
    kw["using_external_codec"]    = False
    kw["vec_qat"]                 = False
    kw["decode_from_latent_code"] = False
    # keep shadingMode consistent with current arg (only if present in ckpt)
    if kw.get("shadingMode", None) is not None and hasattr(args, "shadingMode"):
        kw["shadingMode"] = getattr(args, "shadingMode")
    return kw

# ----------------------------------------------------------------------
# Model loading (no reconstruction / no codec init)
# ----------------------------------------------------------------------
@torch.no_grad()
def load_tensorf_for_eval(args):
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # We require kwargs to rebuild the module (normal for your training/eval saves)
    if not isinstance(ckpt, dict) or "kwargs" not in ckpt:
        raise RuntimeError(
            f"Checkpoint {args.ckpt} does not contain 'kwargs'. "
            "Please pass a checkpoint saved by training/eval (not a raw state_dict)."
        )

    kwargs = _force_no_compression_kwargs(ckpt["kwargs"], args)
    Model  = eval(args.model_name)  # e.g., TensorVMSplit

    tensorf = Model(**kwargs)
    tensorf.load(ckpt)            # load planes, lines, basis, MLP, etc.
    tensorf.mode = "eval"         # just in case your forward checks this flag

    # absolutely do NOT initialize any codec here
    # (no tensorf.init_feat_codec(), no .compress_with_external_codec(), etc.)

    return tensorf

# ----------------------------------------------------------------------
# Evaluation entry point
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_ckpt_direct(args):
    # dataset (test split only)
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split="test", downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray  = args.ndc_ray

    # model
    tensorf = load_tensorf_for_eval(args)


    # ourdir = the parent folder of ckpt
    outdir = str(Path(args.ckpt).parent)
    # os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/imgs_test_all", exist_ok=True)
    with open(f"{outdir}/eval_cfg.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # render & PSNR (no extra reconstruction step, no codec anywhere)
    PSNRs_test = evaluation(
        test_dataset, tensorf, args, renderer, f"{outdir}/imgs_test_all/",
        N_vis=-1 if args.N_vis == -1 else args.N_vis,
        N_samples=-1,  # let evaluation pick default
        white_bg=white_bg, ndc_ray=ndc_ray, device=device,
    )

    psnr_mean = float(np.mean(PSNRs_test))
    with open(f"{outdir}/metrics.json", "w") as f:
        json.dump({"psnr_mean": psnr_mean, "psnr_list": [float(x) for x in PSNRs_test]}, f, indent=2)

    print(f"[DONE] {args.expname} test PSNR: {psnr_mean:.4f} dB (N={len(PSNRs_test)})")
    print(f"[OUT]  Images & metrics saved in: {outdir}")
    return psnr_mean

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _make_argparser():
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset
    ap.add_argument("--dataset_name", default="blender", choices=["blender","llff","tankstemple","nsvf","synthetic"])
    ap.add_argument("--datadir", required=True, help="Dataset root")
    ap.add_argument("--downsample_train", type=float, default=1.0)
    ap.add_argument("--ndc_ray", type=int, default=0)

    # model / ckpt
    ap.add_argument("--model_name", default="TensorVMSplit")
    ap.add_argument("--ckpt", required=True, help="Checkpoint (.pth/.th/.tar) produced by training or the 3-in-1 JPEG script")

    # logging
    ap.add_argument("--basedir", default="./log_eval")
    ap.add_argument("--expname", default="eval_direct")
    ap.add_argument("--add_timestamp", action="store_true")
    ap.add_argument("--add_exp_version", action="store_true")

    # rendering controls
    ap.add_argument("--N_vis", type=int, default=-1, help="-1 renders all test views")
    ap.add_argument("--shadingMode", default="MLP_Fea")  # used only if present in ckpt kwargs

    # seed
    ap.add_argument("--seed", type=int, default=20211202)

    return ap

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    args = _make_argparser().parse_args()
    set_seed(args.seed)
    evaluate_ckpt_direct(args)
