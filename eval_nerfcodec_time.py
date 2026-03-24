import os
import json
import time
import math
import sys
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F

from renderer import OctreeRender_trilinear_fast as renderer
from dataLoader import dataset_dict
from renderer import evaluation
from utils import cal_n_samples
from compressai.ops import compute_padding


"""
python eval_nerfcodec_time.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 100 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --system_ckpt log_2/nerf_chair_384/chair_codec_384_system_19999.th \
    --ckpt log_2/nerf_chair_384/chair_codec_384_compression_19999.th
"""

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

def _alpha_nonempty(tensorf):
    am = getattr(tensorf, "alphaMask", None)
    if am is None:
        return True
    with torch.no_grad():
        vol = am.alpha_volume
        return bool((vol > 0).any().item())

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

# ---------------- latency measurement helpers ----------------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _safe_mean(xs):
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def _safe_std(xs):
    return float(np.std(xs)) if len(xs) > 0 else float("nan")


def _sum_strings_bytes(strings):
    """
    CompressAI-style strings:
      - usually a list of lists, one inner list per entropy stream, one item per batch element
      - for batch size 1, often strings = [[bytes...], [bytes...]]
    """
    total = 0
    for group in strings:
        if isinstance(group, (list, tuple)):
            for s in group:
                total += len(s)
        else:
            total += len(group)
    return int(total)


def _prepare_plane_decode_cache(plane, feat_codec):
    """
    Mirror the eval-time preprocessing in feature_compression_via_feat_coder(),
    but stop after feat_codec.compress(...). This lets us time pure decode later.
    """
    plane = torch.tanh(plane)

    min_vec = plane.amin(dim=(-2, -1), keepdim=True)
    max_vec = plane.amax(dim=(-2, -1), keepdim=True)
    span = max_vec - min_vec
    norm_plane = (plane - min_vec) / span.clamp_min(1e-8)

    h, w = norm_plane.shape[-2:]
    pad, unpad = compute_padding(h, w, min_div=2 ** 6)
    x = F.pad(norm_plane, pad, mode="constant", value=0)

    out_enc = feat_codec.compress(x)

    return {
        "kind": "plane",
        "strings": out_enc["strings"],
        "shape": out_enc["shape"],
        "unpad": unpad,
        "h": h,
        "w": w,
        "min_vec": min_vec,
        "span": span,
        "bitstream_bytes": _sum_strings_bytes(out_enc["strings"]),
    }


def _decode_plane_from_cache(cache, feat_codec):
    out_dec = feat_codec.decompress(cache["strings"], cache["shape"])
    x_hat = F.pad(out_dec["x_hat"], cache["unpad"])
    x_hat = x_hat.reshape(1, -1, cache["h"], cache["w"])
    rec_plane = x_hat * cache["span"] + cache["min_vec"]
    return rec_plane


def _prepare_latent_decode_cache(y, z, feat_codec, target_plane):
    """
    For decode_from_latent_code=True.
    """
    h, w = target_plane.shape[-2:]
    pad, unpad = compute_padding(h, w, min_div=2 ** 6)
    out_enc = feat_codec.compress(y, z)

    return {
        "kind": "latent",
        "strings": out_enc["strings"],
        "shape": out_enc["shape"],
        "unpad": unpad,
        "h": h,
        "w": w,
        "bitstream_bytes": _sum_strings_bytes(out_enc["strings"]),
    }


def _decode_latent_from_cache(cache, feat_codec):
    out_dec = feat_codec.decompress(cache["strings"], cache["shape"])
    x_hat = F.pad(out_dec["x_hat"], cache["unpad"])
    x_hat = x_hat.reshape(1, -1, cache["h"], cache["w"])
    return x_hat


@torch.no_grad()
def build_decode_cache(tensorf, args):
    """
    Build compressed bitstreams once, without including them in decode timing.
    Returns:
      cache: {"den": [...], "app": [...]}
      den_bitstream_bytes
      app_bitstream_bytes
    """
    den_cache, app_cache = [], []
    den_bytes, app_bytes = 0, 0

    if args.decode_from_latent_code:
        for idx in range(len(tensorf.density_plane)):
            c = _prepare_latent_decode_cache(
                tensorf.den_latent_y[idx],
                tensorf.den_latent_z[idx],
                tensorf.den_feat_codec,
                tensorf.density_plane[idx],
            )
            den_cache.append(c)
            den_bytes += c["bitstream_bytes"]

        for idx in range(len(tensorf.app_plane)):
            c = _prepare_latent_decode_cache(
                tensorf.app_latent_y[idx],
                tensorf.app_latent_z[idx],
                tensorf.app_feat_codec,
                tensorf.app_plane[idx],
            )
            app_cache.append(c)
            app_bytes += c["bitstream_bytes"]
    else:
        for idx in range(len(tensorf.density_plane)):
            c = _prepare_plane_decode_cache(
                tensorf.density_plane[idx],
                tensorf.den_feat_codec,
            )
            den_cache.append(c)
            den_bytes += c["bitstream_bytes"]

        for idx in range(len(tensorf.app_plane)):
            c = _prepare_plane_decode_cache(
                tensorf.app_plane[idx],
                tensorf.app_feat_codec,
            )
            app_cache.append(c)
            app_bytes += c["bitstream_bytes"]

    return {"den": den_cache, "app": app_cache}, den_bytes, app_bytes


@torch.no_grad()
def benchmark_decode_from_cache(tensorf, cache, args):
    """
    Time only decoder-side reconstruction from already-built bitstreams.
    Also writes the final decoded planes back into tensorf for later rendering.
    """
    decode_fn = _decode_latent_from_cache if args.decode_from_latent_code else _decode_plane_from_cache

    warmup = max(0, int(args.decode_warmup))
    repeats = max(1, int(args.decode_repeat))

    times = []
    last_den, last_app = None, None

    for rep in range(warmup + repeats):
        _sync()
        t0 = time.perf_counter()

        den_rec = [decode_fn(c, tensorf.den_feat_codec) for c in cache["den"]]
        app_rec = [decode_fn(c, tensorf.app_feat_codec) for c in cache["app"]]

        _sync()
        dt = time.perf_counter() - t0

        last_den, last_app = den_rec, app_rec
        if rep >= warmup:
            times.append(dt)

    tensorf.den_rec_plane = last_den
    tensorf.app_rec_plane = last_app

    mean_sec = _safe_mean(times)
    return {
        "num_density_planes": len(cache["den"]),
        "num_appearance_planes": len(cache["app"]),
        "num_planes_total": len(cache["den"]) + len(cache["app"]),
        "warmup_iters": warmup,
        "timed_iters": len(times),
        "decode_once_sec_mean": mean_sec,
        "decode_once_sec_std": _safe_std(times),
        "decode_fps_scene": (1.0 / mean_sec) if np.isfinite(mean_sec) and mean_sec > 0 else float("nan"),
    }


def _get_timing_view_samples(test_dataset, timing_N_vis):
    if timing_N_vis == 0:
        raise ValueError("--timing_N_vis must be -1 (all) or a positive integer.")

    total = test_dataset.all_rays.shape[0]
    img_eval_interval = 1 if timing_N_vis < 0 else max(total // timing_N_vis, 1)
    idxs = list(range(0, total, img_eval_interval))
    samples = test_dataset.all_rays[0::img_eval_interval]
    return idxs, samples


@torch.no_grad()
def benchmark_render_fps(test_dataset, tensorf, args, white_bg, ndc_ray, device):
    """
    Time only the renderer(...) calls over selected views.
    Excludes PSNR/SSIM/LPIPS/image writing.
    """
    idxs, samples = _get_timing_view_samples(test_dataset, args.timing_N_vis)
    total_views = len(idxs)

    if total_views == 0:
        return {
            "num_views_total_considered": 0,
            "num_warmup_views": 0,
            "num_views_timed": 0,
            "render_total_sec_after_warmup": float("nan"),
            "render_sec_per_view_after_warmup": float("nan"),
            "render_fps_after_warmup": float("nan"),
        }

    # Keep at least one timed view when possible.
    warmup_views = min(max(0, int(args.render_warmup_views)), max(total_views - 1, 0))

    per_view_times = []
    nSamples = tensorf.nSamples

    for i, sample in tqdm(
        enumerate(samples),
        total=total_views,
        desc="Benchmark render FPS",
        file=sys.stdout,
    ):
        rays = sample.reshape(-1, sample.shape[-1])

        _sync()
        t0 = time.perf_counter()
        rgb_map, _, depth_map, _, _ = renderer(
            rays,
            tensorf,
            chunk=args.render_chunk,
            N_samples=nSamples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        _sync()
        dt = time.perf_counter() - t0

        # touch outputs so the timing definitely covers real work
        _ = rgb_map.shape[0] + depth_map.shape[0]

        if i >= warmup_views:
            per_view_times.append(dt)
            if args.render_timed_views > 0 and len(per_view_times) >= args.render_timed_views:
                break

    total_sec = float(np.sum(per_view_times)) if len(per_view_times) > 0 else float("nan")
    sec_per_view = (total_sec / len(per_view_times)) if len(per_view_times) > 0 else float("nan")
    fps = (len(per_view_times) / total_sec) if len(per_view_times) > 0 and total_sec > 0 else float("nan")

    return {
        "num_views_total_considered": total_views,
        "num_warmup_views": warmup_views,
        "num_views_timed": len(per_view_times),
        "render_total_sec_after_warmup": total_sec,
        "render_sec_per_view_after_warmup": sec_per_view,
        "render_fps_after_warmup": fps,
    }

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

    # ---------------- Compression eval branch + speed benchmarks ----------------
    den_bitstream_bytes = 0
    app_bitstream_bytes = 0
    decode_timing = None

    if args.compression:
        tensorf.den_feat_codec.update(force=True)
        tensorf.app_feat_codec.update(force=True)
        tensorf.den_feat_codec.eval()
        tensorf.app_feat_codec.eval()
        tensorf.mode = "eval"

        if args.compress_before_volrend:
            # 1) build compressed strings once (used for exact bitstream bytes)
            decode_cache, den_bitstream_bytes, app_bitstream_bytes = build_decode_cache(tensorf, args)

            # 2) benchmark pure decoder-side reconstruction from those cached strings
            decode_timing = benchmark_decode_from_cache(tensorf, decode_cache, args)
            _assert_nonempty_rec_planes(tensorf)

        else:
            # fallback: no pre-decoded planes; speed split is less meaningful in this mode
            den_like, app_like = tensorf.get_rate()
            den_bits = _rate_from_likelihoods(den_like)
            app_bits = _rate_from_likelihoods(app_like)
            den_bitstream_bytes = int(den_bits.item() / 8.0)
            app_bitstream_bytes = int(app_bits.item() / 8.0)

    # ---------------- PSNR on test set ----------------
    # Use a reasonable number of samples: if you passed a tiny value like 10, upgrade
    nSamples = tensorf.nSamples
    # if ckpt dir is log_2/nerf_chair/chair_codec_system_34999.th, save_dir = log_2/nerf_chair/eval_outputs/
    save_dir = str(Path(args.ckpt).parent / "eval_outputs")

    # ---------------- Render-only speed benchmark ----------------
    render_timing = benchmark_render_fps(
        test_dataset=test_dataset,
        tensorf=tensorf,
        args=args,
        white_bg=white_bg,
        ndc_ray=ndc_ray,
        device=device,
    )

    # ---------------- PSNR on test set ----------------
    nSamples = tensorf.nSamples
    save_dir = str(Path(args.ckpt).parent / "eval_outputs")
    PSNRs_test = evaluation(
        test_dataset,
        tensorf,
        args,
        renderer,
        save_dir,
        N_vis=args.N_vis,
        N_samples=nSamples,
        white_bg=white_bg,
        ndc_ray=ndc_ray,
        device=device,
    )

    # ---------------- Decoder-parameter bandwidth (optional) ----------------
    decoder_payload_raw = None
    decoder_payload_quant = None

    if args.compression and hasattr(tensorf, "estimate_codec_transmission_bits"):
        # raw = exact dtype sizes; quant-ent = uniform q_bits + entropy bound (+ tiny per-tensor header)
        raw_report = tensorf.estimate_codec_transmission_bits(
            mode="raw", return_breakdown=True
        )
        quant_report = tensorf.estimate_codec_transmission_bits(
            mode="quant-ent", q_bits=args.q_bits_est, include_header=True, return_breakdown=True
        )

        def _slim_report(rep: dict):
            # Keep totals and breakdown (both bits and MB) for density/appearance codecs and renderer
            return {
                "density_codec": {
                    "total_bits": float(rep["density_codec"]["total_bits"]),
                    "total_MB": float(rep["density_codec"]["total_MB"]),
                    "breakdown_bits": rep["density_codec"].get("breakdown_bits"),
                    "breakdown_MB": rep["density_codec"].get("breakdown_MB"),
                },
                "appearance_codec": {
                    "total_bits": float(rep["appearance_codec"]["total_bits"]),
                    "total_MB": float(rep["appearance_codec"]["total_MB"]),
                    "breakdown_bits": rep["appearance_codec"].get("breakdown_bits"),
                    "breakdown_MB": rep["appearance_codec"].get("breakdown_MB"),
                },
                "renderer": {
                    "total_bits": float(rep["renderer"]["total_bits"]),
                    "total_MB": float(rep["renderer"]["total_MB"]),
                    "breakdown_bits": rep["renderer"].get("breakdown_bits"),
                    "breakdown_MB": rep["renderer"].get("breakdown_MB"),
                },
                "total_bits_all": float(rep["total_bits_all"]),
                "total_MB_all": float(rep["total_MB_all"]),
            }

        decoder_payload_raw = _slim_report(raw_report)
        decoder_payload_quant = _slim_report(quant_report)

    # ---------------- Print summary ----------------
    os.makedirs(save_dir, exist_ok=True)
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
        "speed": {
            "decode": decode_timing,
            "render": render_timing,
        },
        "decoder_renderer_payload": {
            "raw": decoder_payload_raw,
            f"quant_ent@{args.q_bits_est}bit": decoder_payload_quant,
        },
    }
    print("\n===== Evaluation Summary =====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(save_dir, "eval_summary.json"), "w") as f:
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
    p.add_argument("--N_vis", type=int, default=5, help='N images to vis')

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

    # Speed benchmarking
    p.add_argument("--decode_warmup", type=int, default=1,
                   help="Number of warmup decode iterations before timing.")
    p.add_argument("--decode_repeat", type=int, default=10,
                   help="Number of timed decode iterations.")
    p.add_argument("--timing_N_vis", type=int, default=-1,
                   help="How many test views to use for render FPS (-1 = all test views).")
    p.add_argument("--render_warmup_views", type=int, default=5,
                   help="Number of warmup views before timing render FPS.")
    p.add_argument("--render_timed_views", type=int, default=-1,
                   help="Cap timed render views after warmup (-1 = all remaining).")
    p.add_argument("--render_chunk", type=int, default=4096,
                   help="Chunk size used by renderer during render FPS benchmarking.")

    # Overhead estimation
    p.add_argument("--q_bits_est", type=int, default=8, help="q_bits used in quant-ent overhead estimate")

    return p


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)
    args = build_argparser().parse_args()
    run_eval(args)
