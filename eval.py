import os
import json
import glob
import os
import pdb
import sys
import time
import json
import random
import datetime
import math
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from renderer import OctreeRender_trilinear_fast as renderer

from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ---------------------------------------------------------------------
# Model (resume) build — mirrors training’s resume path
def _resize_eb_buffers_to_ckpt(eb, ckpt_state, prefix):
    """
    Make the target entropy_bottleneck buffers have the same shapes
    as those stored in the checkpoint, so load_state_dict won't fail.

    prefix e.g.: "den_feat_codec.entropy_bottleneck" or "app_feat_codec.entropy_bottleneck"
    """
    # Which buffers to align if present in ckpt
    for name in ["_quantized_cdf", "_offset", "_cdf_length"]:
        k = f"{prefix}.{name}"
        if k in ckpt_state:
            src = ckpt_state[k]
            # Re-register the buffer with the right shape/dtype/device
            # (replace the existing one so load_state_dict sees matching shapes)
            try:
                # Delete old buffer if it exists
                delattr(eb, name)
            except Exception:
                pass
            eb.register_buffer(name, torch.zeros_like(src, device=src.device, dtype=src.dtype))


def build_model_from_ckpt(args):
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt["kwargs"]; kwargs.update({"device": device})
    if args.compression:
        kwargs.update({
            "compression_strategy": args.compression_strategy,
            "compress_before_volrend": args.compress_before_volrend,
        })
        if args.vec_qat: kwargs["vec_qat"] = True
        if args.decode_from_latent_code: kwargs["decode_from_latent_code"] = True
        if kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
            kwargs["shadingMode"] = args.shadingMode

    Model = eval(args.model_name)
    tensorf = Model(**kwargs)

    # 1) Create codec modules FIRST so they exist in the module tree
    if args.compression:
        tensorf.init_feat_codec(
            codec_ckpt_path=args.codec_ckpt,   # usually ""
            adaptor_q_bit=args.adaptor_q_bit,
            codec_backbone_type=args.codec_backbone_type,
        )
        if args.additional_vec:
            tensorf.init_additional_volume(device=device)
        tensorf.enable_vec_qat()

        # 2) Align EB buffer shapes to the ckpt (BOTH den and app codecs)
        sd = ckpt["state_dict"]
        _resize_eb_buffers_to_ckpt(tensorf.den_feat_codec.entropy_bottleneck, sd,
                                   "den_feat_codec.entropy_bottleneck")
        _resize_eb_buffers_to_ckpt(tensorf.app_feat_codec.entropy_bottleneck, sd,
                                   "app_feat_codec.entropy_bottleneck")

    # 3) Now load all finetuned weights (including codec) safely
    tensorf.load(ckpt)

    return tensorf




# ---------------------------------------------------------------------
# Bit-rate helper (same math as training)
# ---------------------------------------------------------------------
def _rate_from_likelihoods(likelihood_list):
    rate = 0
    for pack in likelihood_list:
        rate += sum((torch.log(l).sum() / (-math.log(2))) for l in pack["likelihoods"].values())
    return rate


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def run_eval(args):
    # ---------------- Dataset ----------------
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    # ---------------- Model ----------------
    tensorf = build_model_from_ckpt(args)
    tensorf.to(device)

    # ---------------- Compression eval branch ----------------
    den_bitstream_bytes = 0
    app_bitstream_bytes = 0

    if args.compression:
        # Freeze & update entropy models for eval (exactly like training)
        tensorf.den_feat_codec.update(force=True)
        tensorf.app_feat_codec.update(force=True)
        tensorf.den_feat_codec.eval()
        tensorf.app_feat_codec.eval()
        tensorf.mode = "eval"

        if args.compress_before_volrend:
            # Produce actual bitstreams (eval path)
            coding_output = tensorf.compress_with_external_codec(
                tensorf.den_feat_codec, tensorf.app_feat_codec, mode="eval"
            )
            den_rate_list = coding_output["den"]["rec_likelihood"]
            app_rate_list = coding_output["app"]["rec_likelihood"]
            # In eval path we get true coded length (strings_length)
            den_bitstream_bytes = sum([p["strings_length"] for p in den_rate_list])
            app_bitstream_bytes = sum([p["strings_length"] for p in app_rate_list])
        else:
            # Fallback: estimate via log-likelihoods
            den_like, app_like = tensorf.get_rate()
            den_bits = _rate_from_likelihoods(den_like)
            app_bits = _rate_from_likelihoods(app_like)
            den_bitstream_bytes = int(den_bits.item() / 8.0)
            app_bitstream_bytes = int(app_bits.item() / 8.0)

    # ---------------- PSNR on test set ----------------
    PSNRs_test = evaluation(
        test_dataset,
        tensorf,
        args,
        renderer,
        args.save_dir if args.save_dir else "./eval_outputs/",
        N_vis=-1,
        N_samples=-1,
        white_bg=white_bg,
        ndc_ray=ndc_ray,
        device=device,
    )

    # ---------------- Decoder-parameter bandwidth (our function) ----------------
    decoder_bits_raw = None
    decoder_bits_quant = None
    quant_breakdown = None
    raw_breakdown = None
    if args.compression and hasattr(tensorf, "estimate_codec_transmission_bits"):
        # raw (exact serialization of current dtype)
        raw_report = tensorf.estimate_codec_transmission_bits(
            mode="raw", return_breakdown=True
        )
        decoder_bits_raw = raw_report["total_bits_both_codecs"]
        raw_breakdown = {
            "density": raw_report["density_codec_bits"]["breakdown"],
            "appearance": raw_report["appearance_codec_bits"]["breakdown"],
        }

        # quantized + entropy-coded estimate
        quant_report = tensorf.estimate_codec_transmission_bits(
            mode="quant-ent", q_bits=args.q_bits_est, include_header=True, return_breakdown=True
        )
        decoder_bits_quant = quant_report["total_bits_both_codecs"]
        quant_breakdown = {
            "density": quant_report["density_codec_bits"]["breakdown"],
            "appearance": quant_report["appearance_codec_bits"]["breakdown"],
        }

    # ---------------- Print summary ----------------
    os.makedirs(args.save_dir, exist_ok=True)
    summary = {
        "ckpt": args.ckpt,
        "dataset": args.dataset_name,
        "mean_PSNR": float(np.mean(PSNRs_test)),
        "bitstream_bytes": {
            "density_planes": int(den_bitstream_bytes),
            "appearance_planes": int(app_bitstream_bytes),
            "total": int(den_bitstream_bytes + app_bitstream_bytes),
            "total_MB": (den_bitstream_bytes + app_bitstream_bytes) * 1e-6,
        },
        "decoder_side_overhead_bits": {
            "raw_serialize_bits": float(decoder_bits_raw) if decoder_bits_raw is not None else None,
            "quantized_entropy_bits@{}bit".format(args.q_bits_est): float(decoder_bits_quant) if decoder_bits_quant is not None else None,
            "raw_breakdown": raw_breakdown,
            "quant_breakdown": quant_breakdown,
        },
    }
    print("\n===== Evaluation Summary =====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(args.save_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser("Evaluate a compressed TensorVMSplit checkpoint (PSNR, bitstream size, decoder overhead).")

    # Required-ish
    p.add_argument("--ckpt", type=str, required=True, help="Path to *.th checkpoint")
    p.add_argument("--dataset_name", type=str, default="blender", help="dataset key in dataset_dict (e.g., blender)")
    p.add_argument("--datadir", type=str, required=True, help="dataset root")
    p.add_argument("--downsample_train", type=float, default=1.0)
    p.add_argument("--ndc_ray", type=int, default=0)

    # Model name used in your project
    p.add_argument("--model_name", type=str, default="TensorVMSplit")

    # Compression flags (should mirror training)
    p.add_argument("--compression", action="store_true", default=True)
    p.add_argument("--compression_strategy", type=str, default="adaptor_feat_coding")
    p.add_argument("--compress_before_volrend", action="store_true", default=True)

    # Codec backbone settings (match training defaults)
    p.add_argument("--codec_backbone_type", type=str, default="cheng2020-anchor")
    p.add_argument("--adaptor_q_bit", type=int, default=8)
    p.add_argument("--codec_ckpt", type=str, default="", help="optional path to a codec ckpt; empty = use pretrained base")

    # Misc toggles
    p.add_argument("--vec_qat", action="store_true", default=False)
    p.add_argument("--decode_from_latent_code", action="store_true", default=False)
    p.add_argument("--additional_vec", action="store_true", default=False)
    p.add_argument("--shadingMode", type=str, default="MLP_Fea")

    # Output
    p.add_argument("--save_dir", type=str, default="./eval_outputs/")

    # Overhead estimation
    p.add_argument("--q_bits_est", type=int, default=8, help="q_bits used in quant-ent overhead estimate")

    return p


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)
    args = build_argparser().parse_args()

    """
    python eval.py --ckpt log/lego_codec/version_000/lego_codec_compression.th \
        --dataset_name blender --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/lego --downsample_train 1 \
        --compression --compression_strategy adaptor_feat_coding --compress_before_volrend
    """

    run_eval(args)
