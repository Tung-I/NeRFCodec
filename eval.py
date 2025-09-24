import os
import json
import sys
import math
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm

from renderer import OctreeRender_trilinear_fast as renderer
from dataLoader import dataset_dict
from renderer import evaluation
from utils import cal_n_samples
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


# ---------------- buffer shape alignment helpers ----------------

def _resize_eb_buffers_to_ckpt(eb, ckpt_state, prefix):
    """
    Align EntropyBottleneck buffers to the shapes in `ckpt_state` so load_state_dict won't fail.
    prefix examples:
      "den_feat_codec.entropy_bottleneck"
      "app_feat_codec.entropy_bottleneck"
    """
    for name in ["_quantized_cdf", "_offset", "_cdf_length"]:
        k = f"{prefix}.{name}"
        if k in ckpt_state:
            src = ckpt_state[k]
            try:
                delattr(eb, name)
            except Exception:
                pass
            eb.register_buffer(name, torch.zeros_like(src, device=src.device, dtype=src.dtype))


def _resize_gc_buffers_to_ckpt(gc, ckpt_state, prefix):
    """
    Align GaussianConditional buffers/param to the shapes in `ckpt_state`.
    prefix examples:
      "den_feat_codec.gaussian_conditional"
      "app_feat_codec.gaussian_conditional"
    """
    for name in ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]:
        k = f"{prefix}.{name}"
        if k in ckpt_state:
            src = ckpt_state[k]
            # remove as buffer/param if present
            try:
                delattr(gc, name)
            except Exception:
                pass
            # re-register as buffer (works for modern CompressAI); fallback to Parameter for old versions
            try:
                gc.register_buffer(name, torch.zeros_like(src, device=src.device, dtype=src.dtype))
            except Exception:
                from torch.nn import Parameter
                if name == "scale_table":
                    setattr(gc, name, Parameter(torch.zeros_like(src, device=src.device, dtype=src.dtype), requires_grad=False))
                else:
                    raise


def _maybe_guess_system_ckpt(path: str):
    """If user passes ..._compression_XXXXX.th, try to find sibling ..._system_XXXXX.th."""
    if "_compression_" in path:
        cand = path.replace("_compression_", "_system_")
        if os.path.exists(cand):
            return cand
    return None


# ---------------- model build (mirrors training save/load) ----------------

def build_model_from_ckpt(args):
    """
    Preference order:
      1) args.system_ckpt if provided and exists (full model+codec)
      2) infer sibling *_system_*.th from args.ckpt
      3) fallback: model-only args.ckpt + initialize codec from backbone (args.codec_ckpt)
    """
    from models.tensoRF import TensorVMSplit  # ensure class is registered

    # 1/2) Try system checkpoint first
    system_path = getattr(args, "system_ckpt", "") or ""
    if not system_path:
        system_path = _maybe_guess_system_ckpt(args.ckpt)
        if system_path:
            print(f"[eval] Using inferred system checkpoint: {system_path}")

    if system_path and os.path.exists(system_path):
        system = torch.load(system_path, map_location=device, weights_only=False)
        kwargs = dict(system["kwargs"])
        kwargs.update({"device": device})

        Model = eval(args.model_name)
        tensorf = Model(**kwargs)
        if hasattr(tensorf, "enable_vec_qat"):
            tensorf.enable_vec_qat()
        tensorf.compression = True
        tensorf.compress_before_volrend = True   # matches your recipe      

        if args.compression:
            # Create codec modules (no pretrained), then overwrite from system ckpt
            tensorf.init_feat_codec(
                codec_ckpt_path="",
                loading_pretrain_param=False,
                adaptor_q_bit=args.adaptor_q_bit,
                codec_backbone_type=args.codec_backbone_type,
            )
            sd = system["state_dict"]
            # Align EB + GC buffer shapes before load
            _resize_eb_buffers_to_ckpt(
                tensorf.den_feat_codec.entropy_bottleneck, sd, "den_feat_codec.entropy_bottleneck"
            )
            _resize_eb_buffers_to_ckpt(
                tensorf.app_feat_codec.entropy_bottleneck, sd, "app_feat_codec.entropy_bottleneck"
            )
            _resize_gc_buffers_to_ckpt(
                tensorf.den_feat_codec.gaussian_conditional, sd, "den_feat_codec.gaussian_conditional"
            )
            _resize_gc_buffers_to_ckpt(
                tensorf.app_feat_codec.gaussian_conditional, sd, "app_feat_codec.gaussian_conditional"
            )

        # Load full state (TensoRF + codec + alphaMask)
        tensorf.load(system)  # TensorBase.load uses strict=False and restores alphaMask if present
        return tensorf

    # 3) Fallback: model-only ckpt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = dict(ckpt["kwargs"])
    kwargs.update({"device": device})

    if args.compression:
        kwargs.update({
            "compression_strategy": args.compression_strategy,
            "compress_before_volrend": args.compress_before_volrend,
        })
        if args.vec_qat:
            kwargs["vec_qat"] = True
        if args.decode_from_latent_code:
            kwargs["decode_from_latent_code"] = True
        if kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
            kwargs["shadingMode"] = args.shadingMode

    Model = eval(args.model_name)
    tensorf = Model(**kwargs)

    if args.compression:
        # Initialize codec from a backbone (or default zoo)
        tensorf.init_feat_codec(
            codec_ckpt_path=args.codec_ckpt,
            adaptor_q_bit=args.adaptor_q_bit,
            codec_backbone_type=args.codec_backbone_type,
        )
        # If the model-only ckpt actually contains codec priors, align shapes too
        sd = ckpt["state_dict"]
        _resize_eb_buffers_to_ckpt(
            tensorf.den_feat_codec.entropy_bottleneck, sd, "den_feat_codec.entropy_bottleneck"
        )
        _resize_eb_buffers_to_ckpt(
            tensorf.app_feat_codec.entropy_bottleneck, sd, "app_feat_codec.entropy_bottleneck"
        )
        _resize_gc_buffers_to_ckpt(
            tensorf.den_feat_codec.gaussian_conditional, sd, "den_feat_codec.gaussian_conditional"
        )
        _resize_gc_buffers_to_ckpt(
            tensorf.app_feat_codec.gaussian_conditional, sd, "app_feat_codec.gaussian_conditional"
        )

    # Load model weights (and codec bits if present in ckpt)
    tensorf.load(ckpt)
    return tensorf


# ---------------- rate helper (unchanged) ----------------

def _rate_from_likelihoods(likelihood_list):
    rate = 0
    for pack in likelihood_list:
        rate += sum((torch.log(l).sum() / (-math.log(2))) for l in pack["likelihoods"].values())
    return rate


# ---------------- sanity checks ----------------

def _print_plane_stats(tensorf, title):
    with torch.no_grad():
        def flat_stats(name, planes):
            if not planes:
                print(f"[{title}] {name}: <empty>")
                return
            flat = torch.cat([p.reshape(-1) for p in planes], dim=0)
            mn, mx = flat.min().item(), flat.max().item()
            mu, sd = flat.mean().item(), flat.std(unbiased=False).item()
            print(f"[{title}] {name}: min={mn:.6f} max={mx:.6f} mean={mu:.6f} std={sd:.6f}")

        den = getattr(tensorf, "den_rec_plane", None)
        app = getattr(tensorf, "app_rec_plane", None)
        if den is not None and app is not None:
            flat_stats("den_rec_plane", den)
            flat_stats("app_rec_plane", app)
        else:
            print(f"[{title}] rec_planes not set yet.")


def _assert_nonempty_rec_planes(tensorf):
    den = getattr(tensorf, "den_rec_plane", None)
    app = getattr(tensorf, "app_rec_plane", None)
    assert den is not None and app is not None and len(den) == 3 and len(app) == 3, \
        "Decoded planes are missing. Make sure compress_with_external_codec(..., mode='eval') ran."
    with torch.no_grad():
        dflat = torch.cat([p.reshape(-1) for p in den], dim=0)
        aflat = torch.cat([p.reshape(-1) for p in app], dim=0)
        assert torch.isfinite(dflat).all() and torch.isfinite(aflat).all(), "Non-finite values in decoded planes."
        dspan = (dflat.max() - dflat.min()).item()
        aspan = (aflat.max() - aflat.min()).item()
        assert dspan > 0 or aspan > 0, "Decoded planes have zero dynamic range."


def _alpha_nonempty(tensorf):
    am = getattr(tensorf, "alphaMask", None)
    if am is None:
        return True
    with torch.no_grad():
        vol = am.alpha_volume
        return bool((vol > 0).any().item())


# ---------------- main evaluation ----------------

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

    # Quick sanity on alpha mask
    if not _alpha_nonempty(tensorf):
        print("[WARN] Loaded alphaMask is empty; renders may be blank. Check the checkpoint pairing.")
    else:
        print("[eval] alphaMask OK (non-empty).")

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
            # Important: run decode before rendering
            coding_output = tensorf.compress_with_external_codec(
                tensorf.den_feat_codec, tensorf.app_feat_codec, mode="eval"
            )
            # Basic check the streams are actually coded
            den_rate_list = coding_output["den"]["rec_likelihood"]
            app_rate_list = coding_output["app"]["rec_likelihood"]
            den_bitstream_bytes = sum([p.get("strings_length", 0) for p in den_rate_list])
            app_bitstream_bytes = sum([p.get("strings_length", 0) for p in app_rate_list])

            # Debug stats to ensure decoded planes are sane
            _print_plane_stats(tensorf, "after_decode")
            _assert_nonempty_rec_planes(tensorf)

        else:
            # Fallback: estimate via log-likelihoods
            den_like, app_like = tensorf.get_rate()
            den_bits = _rate_from_likelihoods(den_like)
            app_bits = _rate_from_likelihoods(app_like)
            den_bitstream_bytes = int(den_bits.item() / 8.0)
            app_bitstream_bytes = int(app_bits.item() / 8.0)

    # ---------------- PSNR on test set ----------------
    # Use a reasonable number of samples: if you passed a tiny value like 10, upgrade
    nSamples = tensorf.nSamples

    PSNRs_test = evaluation(
        test_dataset,
        tensorf,
        args,
        renderer,
        args.save_dir if args.save_dir else "./eval_outputs/",
        N_vis=10,
        N_samples=nSamples,
        white_bg=white_bg,
        ndc_ray=ndc_ray,
        device=device,
    )

    # ---------------- Decoder-parameter bandwidth (optional) ----------------
    decoder_bits_raw = None
    decoder_bits_quant = None
    quant_breakdown = None
    raw_breakdown = None
    if args.compression and hasattr(tensorf, "estimate_codec_transmission_bits"):
        raw_report = tensorf.estimate_codec_transmission_bits(
            mode="raw", return_breakdown=True
        )
        decoder_bits_raw = raw_report["total_bits_both_codecs"]
        raw_breakdown = {
            "density": raw_report["density_codec_bits"]["breakdown"],
            "appearance": raw_report["appearance_codec_bits"]["breakdown"],
        }

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
        "system_ckpt": getattr(args, "system_ckpt", ""),
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


# ---------------- CLI ----------------

def build_argparser():
    p = argparse.ArgumentParser("Evaluate a compressed TensorVMSplit checkpoint (PSNR, bitstream size, decoder overhead).")

    # Required-ish
    p.add_argument("--ckpt", type=str, required=True, help="Path to model or system *.th checkpoint")
    p.add_argument("--system_ckpt", type=str, default="", help="Prefer loading this full system checkpoint (incl. codec)")
    p.add_argument("--dataset_name", type=str, default="blender")
    p.add_argument("--datadir", type=str, required=True)
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
    run_eval(args)
