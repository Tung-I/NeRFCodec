import glob
import os
import pdb
import sys
import time
import json
import random
import datetime
import math
import numpy as np
import wandb

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict
# CHANGED: import PlanesCfg (QP-capable) instead of JPEGPlanesCfg
from models.tensorSTE import TensorSTE, PlanesCfg


# ======================================================================================
# Globals
# ======================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast   # keep identical for compatibility

# ======================================================================================
# Logging gradients
# ======================================================================================
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


@torch.no_grad()
def _compute_param_norm_and_grad_stats(
    model,
    sample_k: int = 200000,
):
    """
    Returns:
      dict with global stats and plane-specific norms.
    Notes:
      - Uses sampling for p95/p99 to keep overhead small.
      - Computes norms in float64 for stability.
    """
    total_param_sq = 0.0
    total_grad_sq = 0.0

    # global grad abs stats
    grad_abs_sum = 0.0
    grad_abs_count = 0

    grad_max_abs = 0.0
    grad_nan_inf = 0

    # sample buffer for quantiles
    sample_buf = []

    # plane-specific grad norms
    den_grad_sq = 0.0
    app_grad_sq = 0.0

    for name, p in model.named_parameters():
        if p is None:
            continue

        # param norm
        if p.requires_grad:
            total_param_sq += _safe_float((p.detach().float().pow(2).sum()).item())

        g = p.grad
        if g is None:
            continue

        g_det = g.detach()
        if not torch.isfinite(g_det).all():
            grad_nan_inf += int((~torch.isfinite(g_det)).sum().item())

        g_f = g_det.float()

        # global L2
        total_grad_sq += _safe_float((g_f.pow(2).sum()).item())

        # abs stats
        abs_g = g_f.abs()
        grad_max_abs = max(grad_max_abs, _safe_float(abs_g.max().item()))
        grad_abs_sum += _safe_float(abs_g.sum().item())
        grad_abs_count += abs_g.numel()

        # sample for quantiles
        # (random subset of abs grads; avoids huge flatten)
        if sample_k > 0:
            n = abs_g.numel()
            if n <= sample_k and len(sample_buf) < sample_k:
                sample_buf.append(abs_g.reshape(-1).cpu())
            else:
                # sample a small chunk from this tensor
                # choose m such that total stays around sample_k
                remaining = max(0, sample_k - sum(t.numel() for t in sample_buf))
                if remaining > 0:
                    m = min(remaining, max(1024, sample_k // 20))
                    flat = abs_g.reshape(-1)
                    idx = torch.randint(0, flat.numel(), (m,), device=flat.device)
                    sample_buf.append(flat[idx].cpu())

        # plane-specific grad norms
        if "density_plane" in name:
            den_grad_sq += _safe_float((g_f.pow(2).sum()).item())
        if "app_plane" in name:
            app_grad_sq += _safe_float((g_f.pow(2).sum()).item())

    grad_norm = math.sqrt(max(total_grad_sq, 0.0))
    param_norm = math.sqrt(max(total_param_sq, 0.0))
    grad_mean_abs = (grad_abs_sum / max(1, grad_abs_count))

    # quantiles (p95/p99) from sampled abs grads
    p95 = float("nan")
    p99 = float("nan")
    if len(sample_buf) > 0:
        samples = torch.cat(sample_buf, dim=0)
        # guard: sometimes empty if sample_k==0
        if samples.numel() > 0:
            p95 = _safe_float(torch.quantile(samples, 0.95).item())
            p99 = _safe_float(torch.quantile(samples, 0.99).item())

    out = {
        "param_norm_l2": param_norm,
        "grad_norm_l2": grad_norm,
        "grad_max_abs": grad_max_abs,
        "grad_mean_abs": grad_mean_abs,
        "grad_p95_abs": p95,
        "grad_p99_abs": p99,
        "grad_nan_inf_count": grad_nan_inf,
        "grad_over_param": (grad_norm / (param_norm + 1e-12)),
        "den_grad_norm_l2": math.sqrt(max(den_grad_sq, 0.0)),
        "app_grad_norm_l2": math.sqrt(max(app_grad_sq, 0.0)),
    }
    return out


def _init_grad_log(logdir: str):
    path = os.path.join(logdir, "grad_stats.txt")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                "it\tloss\tmse\tpsnr\t"
                "grad_l2\tparam_l2\tgrad_over_param\t"
                "grad_max\tgrad_mean\tgrad_p95\tgrad_p99\t"
                "den_grad_l2\tapp_grad_l2\t"
                "nan_inf\n"
            )
    return path


def _append_grad_log(path: str, it: int, loss, mse, psnr, stats: dict):
    with open(path, "a") as f:
        f.write(
            f"{it}\t"
            f"{_safe_float(loss)}\t{_safe_float(mse)}\t{_safe_float(psnr)}\t"
            f"{stats['grad_norm_l2']:.6e}\t{stats['param_norm_l2']:.6e}\t{stats['grad_over_param']:.6e}\t"
            f"{stats['grad_max_abs']:.6e}\t{stats['grad_mean_abs']:.6e}\t{stats['grad_p95_abs']:.6e}\t{stats['grad_p99_abs']:.6e}\t"
            f"{stats['den_grad_norm_l2']:.6e}\t{stats['app_grad_norm_l2']:.6e}\t"
            f"{int(stats['grad_nan_inf_count'])}\n"
        )


# ======================================================================================
# Utilities
# ======================================================================================

def _make_model_ckpt_dict(tensorf):
    kwargs = tensorf.get_kwargs()
    ckpt = {"kwargs": kwargs, "state_dict": tensorf.state_dict()}
    if getattr(tensorf, "alphaMask", None) is not None:
        alpha_volume = tensorf.alphaMask.alpha_volume.bool().cpu().numpy()
        ckpt["alphaMask.shape"] = alpha_volume.shape
        ckpt["alphaMask.mask"]  = np.packbits(alpha_volume.reshape(-1))
        ckpt["alphaMask.aabb"]  = tensorf.alphaMask.aabb.cpu()
    return ckpt

def _save_system_ckpt(path, tensorf, optimizer, aux_optimizer, global_step, kwargs_override=None):
    base = _make_model_ckpt_dict(tensorf)
    if kwargs_override is not None:
        base["kwargs"] = kwargs_override
    base["optimizer"]     = optimizer.state_dict()     if optimizer     is not None else None
    base["aux_optimizer"] = aux_optimizer.state_dict() if aux_optimizer is not None else None
    base["global_step"]   = int(global_step)
    torch.save(base, path)

def set_seed(seed: int = 20211202):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleSampler:
    def __init__(self, total: int, batch: int):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None
    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]


def _build_log_dir(args) -> str:
    if args.add_timestamp:
        return f"{args.basedir}/{args.expname}/{datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')}"
    if args.add_exp_version:
        root = f"{args.basedir}/{args.expname}"
        os.makedirs(root, exist_ok=True)
        versions = sorted(glob.glob(f"{root}/version_*"))
        idx = 0 if not versions else (int(versions[-1].split('_')[-1]) + 1)
        return f"{root}/version_{idx:03d}"
    return f"{args.basedir}/{args.expname}"


def _derive_schedule_lists(args):
    upsamp = args.upsamp_list
    updateA = args.update_AlphaMask_list
    if getattr(args, "compression", False):
        upsamp = [100001]
        updateA = [100001]
    return upsamp, updateA


# ======================================================================================
# Model build
# ======================================================================================

# CHANGED: build generalized PlanesCfg with QP params for video codecs
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

         # NEW: grad surrogate
        grad_surrogate_mode=str(getattr(args, "grad_surrogate_mode", "ste")).lower(),
        grad_surrogate_std_eps=float(getattr(args, "grad_surrogate_std_eps", 1e-8)),
        spsa_n_samples=int(getattr(args, "spsa_n_samples", 1)),
        spsa_gate_on_cache_refresh=int(getattr(args, "spsa_gate_on_cache_refresh", 1)),

    )


def _build_model(args, aabb, reso_cur, near_far):
    # CHANGED: enable TensorSTE for jpeg/png/hevc/av1/vp9
    using_tensorste = (str(getattr(args, "codec_backend", "jpeg")).lower()
                       in ("jpeg", "png", "hevc", "av1", "vp9"))
    Model = TensorSTE if using_tensorste else eval(args.model_name)

    # --------- resume (pretrained TensoRF -> finetune) ----------
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        kwargs = ckpt["kwargs"]; kwargs.update({"device": device})
        if args.compression:
            kwargs.update({"compress_before_volrend": True})
            if kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
                kwargs["shadingMode"] = args.shadingMode

        tensorf = Model(**kwargs)
        tensorf.load(ckpt)

        if using_tensorste:
            cfg = _build_planescfg_from_args(args)
            tensorf.init_ste(cfg)
            tensorf.set_ste(bool(getattr(args, "ste_enabled", 1)))
            tensorf.enable_vec_qat()
        else:
            raise Exception("Legacy adaptor path disabled in this script.")
        return tensorf

    # --------- train from scratch ----------
    tensorf = Model(
        aabb, reso_cur, device,
        density_n_comp=args.n_lamb_sigma,
        appearance_n_comp=args.n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingMode,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
        featureC=args.featureC, step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )

    if using_tensorste:
        cfg = _build_planescfg_from_args(args)
        tensorf.init_ste(cfg)
        tensorf.set_ste(bool(getattr(args, "ste_enabled", 1)))
        tensorf.enable_vec_qat()
    else:
        raise Exception("Legacy adaptor path disabled in this script.")
    return tensorf


def _configure_optimizers(tensorf, args):
    if not args.compression:
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        return torch.optim.Adam(grad_vars, betas=(0.9, 0.99)), None

    # STE path: optimize only TensoRF (plus optional extras)
    if not getattr(args, "resume_finetune", 0):
        grad_vars = tensorf.get_optparam_groups(
            lr_init_spatialxyz=2e-3, lr_init_network=1e-4, fix_plane=args.fix_triplane
        )
    else:
        grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3, lr_init_network=0)

    if getattr(args, "additional_vec", 0):
        grad_vars += tensorf.get_additional_optparam_groups(lr_init_spatialxyz=2e-3)

    main_opt = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    return main_opt, None


# ======================================================================================
# Training
# ======================================================================================

def reconstruction(args):
    # -------------------- W&B init --------------------
    logdir = _build_log_dir(args)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(f"{logdir}/imgs_vis", exist_ok=True)
    with open(f"{logdir}/train_cfg.json", "w") as f:
        json.dump(vars(args), f)

    wandb.init(
        project=args.wandb_project,
        name=args.expname,
        dir=logdir,
        config=vars(args),
        mode=("disabled" if getattr(args, "wandb_off", 0) else "online"),
    )

    # ---- gradient logging ----
    grad_log_every = int(getattr(args, "grad_log_every", 10))     # default: every 10 iters
    grad_sample_k  = int(getattr(args, "grad_sample_k", 200000))  # default: sample size for quantiles
    grad_log_path  = _init_grad_log(logdir)

    # -------------------- dataset --------------------
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split="train", downsample=args.downsample_train, is_stack=False)
    test_dataset  = dataset(args.datadir, split="test",  downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray  = args.ndc_ray

    # -------------------- schedules --------------------
    upsamp_list, update_AlphaMask_list = _derive_schedule_lists(args)

    # -------------------- model --------------------
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    tensorf = _build_model(args, aabb, reso_cur, near_far)

    # wire refresh_k/refresh_eps to TensorSTE cache
    if hasattr(tensorf, "set_codec_cache"):
        tensorf.set_codec_cache(
            refresh_k=args.refresh_k,
            refresh_eps=args.refresh_eps,
            bpp_refresh_k=args.refresh_k,  # match your choice
        )

    # nSamples policy
    if args.compression:
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # optimizers
    optimizer, aux_optimizer = _configure_optimizers(tensorf, args)

    # optional feature reconstruction reference
    if getattr(args, "feat_rec_loss", 0):
        tensorf.copy_pretrain_feats()

    # LR decay
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    # -------------------- rays & sampler --------------------
    torch.cuda.empty_cache()
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    sampler = SimpleSampler(allrays.shape[0], args.batch_size)

    # Regularizer weights
    Ortho_w = args.Ortho_weight
    L1_w    = args.L1_weight_inital
    L1_w_app = args.L1_weight_app
    TV_w_d, TV_w_a = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()

    PSNRs, PSNRs_test = [], [0.0]
    final_it = 0

    # resume global step if you load a system ckpt with states (optional)
    start_iter = 0
    extra_iters = getattr(args, "extra_iters", 0) or args.n_iters
    end_iter = start_iter + extra_iters
    print(f"[resume] starting at iter={start_iter}, running to {end_iter}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for it in pbar:
        final_it = it
        ray_idx = sampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # pre-render compression (our STE codec)
        coding_output = None
        if args.compression and args.compress_before_volrend:
            coding_output = tensorf.compress_with_external_codec(mode="train")

        # render
        rgb_map, _, depth_map, _, _ = renderer(
            rays_train, tensorf, chunk=32768, N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True,
        )
        del depth_map
        torch.cuda.empty_cache()

        # bits logging (six streams)
        log_bits = {}
        if args.compression and coding_output is not None:
            den_packs = coding_output["den"]["rec_likelihood"]
            app_packs = coding_output["app"]["rec_likelihood"]
            for i, pack in enumerate(den_packs):
                log_bits[f"bits/den_{i}"] = int(pack["bits"])
            for i, pack in enumerate(app_packs):
                log_bits[f"bits/app_{i}"] = int(pack["bits"])
            log_bits["bits/den_total"] = sum(log_bits[f"bits/den_{i}"] for i in range(len(den_packs)))
            log_bits["bits/app_total"] = sum(log_bits[f"bits/app_{i}"] for i in range(len(app_packs)))
            log_bits["bits/total"]     = log_bits["bits/den_total"] + log_bits["bits/app_total"]

        # reconstruction + regs
        mse = torch.mean((rgb_map - rgb_train) ** 2)
        loss = mse

        if Ortho_w > 0:
            loss_reg = tensorf.vector_comp_diffs()
            loss += Ortho_w * loss_reg
            wandb.log({"train/reg": float(loss_reg)}, step=it)
        if L1_w > 0:
            loss_l1 = tensorf.density_L1()
            loss += L1_w * loss_l1
            wandb.log({"train/reg_l1": float(loss_l1)}, step=it)
        if L1_w_app > 0:
            loss_l1_app = tensorf.app_L1()
            loss += L1_w_app * loss_l1_app
            wandb.log({"train/reg_l1_app": float(loss_l1_app)}, step=it)
        if TV_w_d > 0:
            TV_w_d *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_w_d
            loss += loss_tv
            wandb.log({"train/reg_tv_density": float(loss_tv)}, step=it)
        if TV_w_a > 0:
            TV_w_a *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_w_a
            loss += loss_tv
            wandb.log({"train/reg_tv_app": float(loss_tv)}, step=it)
        if getattr(args, "feat_rec_loss", 0):
            feat_rec = features_rec_loss(tensorf, coding_output)
            loss += 1e-2 * feat_rec
            wandb.log({"train/feat_rec_loss": float(feat_rec)}, step=it)


        optimizer.zero_grad()
        loss.backward()

        psnr = -10.0 * np.log(mse.detach().item()) / np.log(10.0)

        # ---- gradient stats (log BEFORE optimizer.step) ----
        if (it % grad_log_every) == 0:
            gstats = _compute_param_norm_and_grad_stats(tensorf, sample_k=grad_sample_k)
            _append_grad_log(
                grad_log_path,
                it=it,
                loss=loss.detach().item(),
                mse=mse.detach().item(),
                psnr=psnr,  # you compute psnr just after step currently; move psnr calc earlier or compute here
                stats=gstats,
            )
            # optional: also send to wandb for convenience
            wandb.log({
                "grad/grad_norm_l2": gstats["grad_norm_l2"],
                "grad/param_norm_l2": gstats["param_norm_l2"],
                "grad/grad_over_param": gstats["grad_over_param"],
                "grad/grad_max_abs": gstats["grad_max_abs"],
                "grad/grad_mean_abs": gstats["grad_mean_abs"],
                "grad/grad_p95_abs": gstats["grad_p95_abs"],
                "grad/grad_p99_abs": gstats["grad_p99_abs"],
                "grad/den_grad_norm_l2": gstats["den_grad_norm_l2"],
                "grad/app_grad_norm_l2": gstats["app_grad_norm_l2"],
                "grad/nan_inf_count": gstats["grad_nan_inf_count"],
            }, step=it)

        optimizer.step()

        # metrics & lr decay logging
        PSNRs.append(psnr)
        log = {"train/PSNR": float(psnr), "train/mse": float(mse.detach().item())}
        log.update(log_bits)
        # LR decay
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * lr_factor
        log["lr"] = float(optimizer.param_groups[0]["lr"])
        wandb.log(log, step=it)

        if it % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"it {it:05d} | train_psnr {np.mean(PSNRs):.2f} | test_psnr {np.mean(PSNRs_test):.2f} | mse {mse:.6f}"
            )
            PSNRs = []

        # periodic visualization on test set
        if it % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            if args.compression:
                with torch.no_grad():
                    _ = tensorf.compress_with_external_codec(mode="eval")
                    PSNRs_test = evaluation(
                        test_dataset, tensorf, args, renderer, f"{logdir}/imgs_vis/",
                        N_vis=args.N_vis, prtx=f"{args.codec_backend}{it:06d}_", N_samples=nSamples,
                        white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                    )
                    wandb.log({"test/PSNR": float(np.mean(PSNRs_test))}, step=it)
            else:
                PSNRs_test = evaluation(
                    test_dataset, tensorf, args, renderer, f"{logdir}/imgs_vis/",
                    N_vis=args.N_vis, prtx=f"{it:06d}_", N_samples=nSamples,
                    white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                )
                wandb.log({"test/PSNR": float(np.mean(PSNRs_test))}, step=it)

        # AlphaMask update / upsample schedules
        if it in update_AlphaMask_list:
            if tensorf.gridSize[0] * tensorf.gridSize[1] * tensorf.gridSize[2] < 256 ** 3:
                reso_mask = tensorf.gridSize
            tensorf.alphaMask_offset = 1e-3 if (2000 < it < 10000) else 0
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if it == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                L1_w = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_w)
            if not ndc_ray and it == update_AlphaMask_list[1]:
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                sampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if it in upsamp_list:
            n_vox_next = torch.round(torch.exp(torch.linspace(
                np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1
            ))).long().tolist()[1:][0]
            reso_cur = N_to_reso(n_vox_next, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)
            # reset / scaled LR
            if args.lr_upsample_reset:
                lr_scale = 1.0
                print("reset lr to initial")
            else:
                lr_scale = args.lr_decay_target_ratio ** (it / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # -------------------- save periodic --------------------
        if (final_it + 1) % args.save_every == 0:
            tensorf.save(f"{logdir}/{args.expname}_compression_{final_it}.th")
            save_path = f"{logdir}/{args.expname}_system_{final_it}.th"
            kwargs = tensorf.get_kwargs()
            _save_system_ckpt(save_path, tensorf, optimizer, aux_optimizer, final_it, kwargs)

    # -------------------- final evals --------------------
    if args.render_test:
        os.makedirs(f"{logdir}/imgs_test_all", exist_ok=True)
        if args.compression:
            out = tensorf.compress_with_external_codec(mode="eval")
            den_bits = sum([p["bits"] for p in out["den"]["rec_likelihood"]])
            app_bits = sum([p["bits"] for p in out["app"]["rec_likelihood"]])
            print(f"====> Final {args.codec_backend.upper()} size (bits): {(den_bits+app_bits):.0f}")
            wandb.log({
                "final/bits_den_total": int(den_bits),
                "final/bits_app_total": int(app_bits),
                "final/bits_total":     int(den_bits + app_bits),
            })
        PSNRs_test = evaluation(
            test_dataset, tensorf, args, renderer, f"{logdir}/imgs_test_all/",
            N_vis=-1, N_samples=10, white_bg=white_bg, ndc_ray=ndc_ray, device=device,
        )
        wandb.log({"final/test_PSNR_all": float(np.mean(PSNRs_test))}, step=final_it)
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <======")

    if args.render_train:
        os.makedirs(f"{logdir}/imgs_train_all", exist_ok=True)
        train_eval = dataset(args.datadir, split="train", downsample=args.downsample_train, is_stack=True)
        _ = evaluation(
            train_eval, tensorf, args, renderer, f"{logdir}/imgs_train_all/",
            N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device
        )


# ======================================================================================
# Main
# ======================================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)
    args = config_parser()
    print(args)
    reconstruction(args)
