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

import torch
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm.auto import tqdm

from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict
from utils import maybe_align_cdf_and_load_system, load_optim_states_if_any, save_system_ckpt


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

def _wandb_init(args, logdir):
    project = getattr(args, "wandb_project", "nerfcodec-adaptor")
    run_name = getattr(args, "wandb_run_name", args.expname)
    mode = getattr(args, "wandb_mode", "online")  # or "offline"
    wandb.init(
        project=project,
        name=run_name,
        dir=logdir,
        mode=mode,
        config=vars(args),
    )

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
    """Random permutation mini-batch sampler over a fixed ray set."""
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
        idx = 0 if not versions else (int(versions[-1].split("_")[-1]) + 1)
        return f"{root}/version_{idx:03d}"
    return f"{args.basedir}/{args.expname}"


def _derive_schedule_lists(args):
    upsamp = args.upsamp_list
    updateA = args.update_AlphaMask_list
    # under adaptor finetune we freeze spatial refinement (NeRFCodec behavior)
    if getattr(args, "compression", False):
        upsamp = [100001]
        updateA = [100001]
    return upsamp, updateA


def _rate_from_likelihoods(likelihood_list):
    # identical to original; used when adaptor codec exposes likelihoods
    rate = 0
    for pack in likelihood_list:
        rate += sum((torch.log(l).sum() / (-math.log(2))) for l in pack["likelihoods"].values())
    return rate


def _codec_warmup_if_needed(tensorf, args, logdir):
    if not args.warm_up:
        return
    if args.warm_up_ckpt != "":
        # load pre-warmed adaptors
        tensorf.den_feat_codec.load_state_dict(
            torch.load(f"{args.warm_up_ckpt}/den_feat_codec_{args.warm_up_iters}.pth", weights_only=False)
        )
        tensorf.app_feat_codec.load_state_dict(
            torch.load(f"{args.warm_up_ckpt}/app_feat_codec_{args.warm_up_iters}.pth", weights_only=False)
        )
        print("Loaded warm-up codec weights.")
        return

    print("Warm-up adaptor (only) ...")
    codec_grads, _ = tensorf.get_optparam_from_feat_codec(
        args.lr_feat_codec, fix_decoder_prior=args.fix_decoder_prior, fix_encoder_prior=True
    )
    opt = torch.optim.Adam(codec_grads, betas=(0.9, 0.99))
    os.makedirs(f"{logdir}/warm_up_feat", exist_ok=True)

    for it in tqdm(range(args.warm_up_iters + 1)):
        out = tensorf.compress_with_external_codec(tensorf.den_feat_codec, tensorf.app_feat_codec, mode="train")
        loss = features_rec_loss(tensorf, out)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it == args.warm_up_iters:
            torch.save(tensorf.den_feat_codec.state_dict(), f"{logdir}/den_feat_codec_{it}.pth")
            torch.save(tensorf.app_feat_codec.state_dict(), f"{logdir}/app_feat_codec_{it}.pth")

    del opt
    torch.cuda.empty_cache()


# ======================================================================================
# Model build
# ======================================================================================

def _build_model(args, aabb, reso_cur, near_far):
    """
    Only two modes are supported (the ones you actually run):
      1) Pretrain from scratch (no --ckpt, no --compression)
      2) Resume from ckpt and finetune with --compression and --compression_strategy adaptor_feat_coding
    """
    Model = eval(args.model_name)  # TensorVMSplit only in your configs

    # --------- FULL RESUME: load previous system ckpt (model+codec) ----------
    if getattr(args, "resume_system_ckpt", None):
        assert args.ckpt is None, "When resuming from a full system checkpoint, do not set --ckpt."
        # Build a minimal skeleton first (same kwargs as the saved run)
        system_meta = torch.load(args.resume_system_ckpt, map_location=device, weights_only=False)
        kwargs = system_meta["kwargs"]; kwargs.update({"device": device})

        # Force flags consistent with the saved run (esp. compression & strategy)
        kwargs["compression_strategy"] = kwargs.get("compression_strategy", "adaptor_feat_coding")
        kwargs["compress_before_volrend"] = kwargs.get("compress_before_volrend", True)

        Model = eval(args.model_name)
        tensorf = Model(**kwargs)

        # Initialize codec *modules* so state_dict keys exist (DO NOT warm or overwrite)
        tensorf.init_feat_codec(
            codec_ckpt_path="",              # we will overwrite from system ckpt next
            loading_pretrain_param=False,    # <--- critical: don't pull zoo weights here
            adaptor_q_bit=args.adaptor_q_bit,
            codec_backbone_type=args.codec_backbone_type,
        )

        # Now load the entire system state (TensoRF + codec + priors)
        maybe_align_cdf_and_load_system(tensorf, args.resume_system_ckpt)
        tensorf.enable_vec_qat()

        return tensorf


    # --------- resume path (pretrained TensoRF -> codec finetune) ----------
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})

        # align finetune flags
        if args.compression:
            kwargs.update({"compression_strategy": args.compression_strategy})
            kwargs.update({"compress_before_volrend": args.compress_before_volrend})
            if args.vec_qat:
                kwargs.update({"vec_qat": True})
            if args.decode_from_latent_code:
                kwargs.update({"decode_from_latent_code": True})
            if kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
                kwargs["shadingMode"] = args.shadingMode

        tensorf = Model(**kwargs)
        tensorf.load(ckpt)

        # init adaptor feature codec if requested
        if args.compression:
            assert args.compression_strategy == "adaptor_feat_coding", \
                "This simplified train.py only supports adaptor_feat_coding under compression."
            tensorf.init_feat_codec(
                args.codec_ckpt,
                adaptor_q_bit=args.adaptor_q_bit,
                codec_backbone_type=args.codec_backbone_type,
            )
            if args.joint_train_from_scratch:
                tensorf.init_svd_volume(res=tensorf.gridSize, device=tensorf.device)
            if args.resume_finetune and args.system_ckpt:
                system = torch.load(args.system_ckpt, map_location=device, weights_only=False)
                # align prior CDF shapes then load system weights
                app_cdf = system["state_dict"]["app_feat_codec.entropy_bottleneck._quantized_cdf"].size()
                den_cdf = system["state_dict"]["den_feat_codec.entropy_bottleneck._quantized_cdf"].size()
                tensorf.app_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(app_cdf)
                tensorf.den_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(den_cdf)
                tensorf.load(system)
            if args.additional_vec:
                tensorf.init_additional_volume(device=device)
            tensorf.enable_vec_qat()

        return tensorf

    # --------- train from scratch (pre-train) ----------
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
    return tensorf


def _configure_optimizers(tensorf, args):
    """
    Exactly two cases:
      - no compression: optimize TensoRF only
      - adaptor_feat_coding: optimize TensoRF + (adaptor + prior), plus aux optimizer
    """
    if not args.compression:
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        return torch.optim.Adam(grad_vars, betas=(0.9, 0.99)), None

    # finetune with adaptor codec
    if not args.resume_finetune:
        grad_vars = tensorf.get_optparam_groups(
            lr_init_spatialxyz=2e-3, lr_init_network=1e-4, fix_plane=args.fix_triplane
        )
    else:
        grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3, lr_init_network=0)

    codec_grad, aux_grad = tensorf.get_optparam_from_feat_codec(
        args.lr_feat_codec,
        fix_decoder_prior=args.fix_decoder_prior,
        fix_encoder_prior=getattr(args, "fix_encoder_prior", False),
    )
    grad_vars += codec_grad

    if args.additional_vec:
        grad_vars += tensorf.get_additional_optparam_groups(lr_init_spatialxyz=2e-3)
    if args.decode_from_latent_code:
        grad_vars += tensorf.get_latent_code_groups(lr_latent_code=args.lr_latent_code)

    main_opt = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    _, aux_opt = configure_optimizers(tensorf, args)   # from utils (unchanged)
    return main_opt, aux_opt


# ======================================================================================
# Training
# ======================================================================================

def reconstruction(args):
    # -------------------- dataset --------------------
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split="train", downsample=args.downsample_train, is_stack=False)
    test_dataset  = dataset(args.datadir, split="test",  downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray  = args.ndc_ray

    # -------------------- schedules & logdir --------------------
    upsamp_list, update_AlphaMask_list = _derive_schedule_lists(args)
    logdir = _build_log_dir(args)
    print(f"[logdir] {logdir}")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(f"{logdir}/imgs_vis", exist_ok=True)
    with open(f"{logdir}/train_cfg.json", "w") as f:
        json.dump(vars(args), f)

    _wandb_init(args, logdir)

    # ---- gradient logging ----
    grad_log_every = int(getattr(args, "grad_log_every", 10))       # log every N iters
    grad_sample_k  = int(getattr(args, "grad_sample_k", 200000))   # samples for p95/p99
    grad_log_path  = _init_grad_log(logdir)

    # -------------------- model --------------------
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    tensorf = _build_model(args, aabb, reso_cur, near_far)

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

    # learning-rate decay (with your codec run: target_ratio=1 ⇒ no decay)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    # warm-up adaptors if requested
    if getattr(args, "resume_system_ckpt", None):
        print("[resume] Skipping warm-up; using codec from resume checkpoint.")
    else:
        _codec_warmup_if_needed(tensorf, args, logdir)

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

    start_iter = 0
    if getattr(args, "resume_system_ckpt", None):
        # load optimizer states (optional)
        if getattr(args, "resume_optim", 0):
            system = torch.load(args.resume_system_ckpt, map_location=device, weights_only=False)
            start_iter = system.get("global_step", 0)
            load_optim_states_if_any(system, optimizer, aux_optimizer)

    # how many more steps to run
    extra_iters = getattr(args, "extra_iters", 0) or args.n_iters
    end_iter = start_iter + extra_iters
    print(f"[resume] starting at iter={start_iter}, running to {end_iter}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

    def _psnr_from_mse(mse_val):
        return -10.0 * np.log(mse_val) / np.log(10.0)
    train_psnrs_since_vis = []

    for it in pbar:
        final_it = it
        ray_idx = sampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # pre-render compression (adaptor path) if requested
        if args.compression and args.compress_before_volrend:
            if args.decode_from_latent_code:
                coding_output = tensorf.decode_all_planes(mode="train")
            else:
                coding_output = tensorf.compress_with_external_codec(
                    tensorf.den_feat_codec, tensorf.app_feat_codec, mode="train"
                )
        else:
            coding_output = None

        # render
        rgb_map, _, depth_map, _, _ = renderer(
            rays_train, tensorf, chunk=32768, N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True,
        )
        del depth_map
        torch.cuda.empty_cache()

        # (optional) rate loss from adaptor codec
        if args.compression:
            if args.compress_before_volrend:
                den_like = coding_output["den"]["rec_likelihood"]
                app_like = coding_output["app"]["rec_likelihood"]
            else:
                den_like, app_like = tensorf.get_rate()
            den_rate = _rate_from_likelihoods(den_like)
            app_rate = _rate_from_likelihoods(app_like)
            rate_loss = args.den_rate_weight * den_rate + args.app_rate_weight * app_rate
        else:
            rate_loss = 0.0

        # reconstruction loss + regs
        mse = torch.mean((rgb_map - rgb_train) ** 2)
        loss = mse
        loss_reg_val = 0.0; loss_l1_val = 0.0
        reg_tv_density_val = 0.0; reg_tv_app_val = 0.0
        
        if Ortho_w > 0:
            loss_reg = tensorf.vector_comp_diffs()
            loss += Ortho_w * loss_reg
            loss_reg_val = float(loss_reg.detach().item())
            
        if L1_w > 0:
            loss_l1 = tensorf.density_L1()
            loss += L1_w * loss_l1
            loss_l1_val = float(loss_l1.detach().item())

        if L1_w_app > 0:
            loss_l1_app = tensorf.app_L1()
            loss += L1_w_app * loss_l1_app
            loss_appl1_val = float(loss_l1_app.detach().item())

        if TV_w_d > 0:
            TV_w_d *= lr_factor
            loss_tv_d = tensorf.TV_loss_density(tvreg) * TV_w_d
            loss += loss_tv_d
            reg_tv_density_val = float(loss_tv_d.detach().item())

        if TV_w_a > 0:
            TV_w_a *= lr_factor
            loss_tv_a = tensorf.TV_loss_app(tvreg) * TV_w_a
            loss += loss_tv_a
            reg_tv_app_val = float(loss_tv_a.detach().item())

        if args.compression and args.rate_penalty:
            loss = loss + rate_loss * 1e-9

        if getattr(args, "feat_rec_loss", 0):
            feat_rec = features_rec_loss(tensorf, coding_output)
            loss += 1e-2 * feat_rec


        optimizer.zero_grad()
        if args.compression and aux_optimizer is not None:
            aux_optimizer.zero_grad()

        loss.backward()

        # ---- gradient stats (log BEFORE any optimizer.step) ----
        if (it % grad_log_every) == 0:
            # compute psnr from mse (cheap) for the txt log line
            mse_val_for_psnr = float(mse.detach().item())
            psnr_for_log = -10.0 * np.log(mse_val_for_psnr) / np.log(10.0)

            gstats = _compute_param_norm_and_grad_stats(tensorf, sample_k=grad_sample_k)
            _append_grad_log(
                grad_log_path,
                it=int(it),
                loss=float(loss.detach().item()),
                mse=mse_val_for_psnr,
                psnr=float(psnr_for_log),
                stats=gstats,
            )

            # also log to wandb (so you can compare runs easily)
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
            }, step=int(it))

        optimizer.step()

        # aux optimizer for entropy model (keep AFTER grad logging)
        if args.compression and aux_optimizer is not None:
            aux_loss = tensorf.get_aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        # # aux optimizer for entropy model
        # if args.compression:
        #     aux_optimizer.zero_grad()
        #     aux_loss = tensorf.get_aux_loss()
        #     aux_loss.backward()
        #     aux_optimizer.step()

        mse_val = float(mse.detach().item())
        train_psnr = float(_psnr_from_mse(mse_val))
        train_psnrs_since_vis.append(train_psnr)

        # log per-iter train metrics
        log_dict = {
            "iter": int(it),
            "train/psnr": train_psnr,
            "train/rate_loss": float(rate_loss) if isinstance(rate_loss, torch.Tensor) else float(rate_loss),
        }
        if loss_reg_val is not None:       log_dict["train/loss_reg"] = loss_reg_val
        if loss_l1_val is not None:        log_dict["train/loss_l1"] = loss_l1_val
        # if loss_appl1_val is not None:     log_dict["train/loss_appl1"] = loss_appl1_val
        if reg_tv_density_val is not None: log_dict["train/reg_tv_density"] = reg_tv_density_val
        if reg_tv_app_val is not None:     log_dict["train/reg_tv_app"] = reg_tv_app_val
        wandb.log(log_dict, step=it)


        # metrics & lr decay logging
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * lr_factor

        PSNRs.append(-10.0 * np.log(mse.detach().item()) / np.log(10.0))
        if it % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"it {it:05d} | train_psnr {np.mean(PSNRs):.2f} | test_psnr {np.mean(PSNRs_test):.2f} | mse {mse:.6f}"
            )
            PSNRs = []

        # periodic visualization on test set
        if it % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            if args.compression:
                with torch.no_grad():
                    tensorf.den_feat_codec.update(force=True)
                    tensorf.app_feat_codec.update(force=True)
                    tensorf.den_feat_codec.eval()
                    tensorf.app_feat_codec.eval()
                    if args.compress_before_volrend:
                        _ = tensorf.compress_with_external_codec(
                            tensorf.den_feat_codec.eval(), tensorf.app_feat_codec.eval(), mode="eval"
                        )
                    PSNRs_test = evaluation(
                        test_dataset, tensorf, args, renderer, f"{logdir}/imgs_vis/",
                        N_vis=args.N_vis, prtx=f"compress{it:06d}_", N_samples=nSamples,
                        white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                    )
                    wandb.log({"test/psnr": float(np.mean(PSNRs_test)), "iter": int(it)}, step=it)
                    tensorf.den_feat_codec.train(); tensorf.app_feat_codec.train()
            else:
                PSNRs_test = evaluation(
                    test_dataset, tensorf, args, renderer, f"{logdir}/imgs_vis/",
                    N_vis=args.N_vis, prtx=f"{it:06d}_", N_samples=nSamples,
                    white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                )
                wandb.log({"test/psnr": float(np.mean(PSNRs_test)), "iter": int(it)}, step=it)

            # (optional) also log mean train psnr since last vis
            if len(train_psnrs_since_vis) > 0:
                wandb.log({"train/psnr@vis": float(np.mean(train_psnrs_since_vis)), "iter": int(it)}, step=it)
                train_psnrs_since_vis.clear()

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

        # -------------------- save final --------------------
        if (final_it+1) % args.save_every == 0:
            if args.compression:
                # Save codec modules and planes
                tensorf.save(f"{logdir}/{args.expname}_compression_{final_it}.th")

                # Save a full system checkpoint for exact resumption of training
                save_path = f"{logdir}/{args.expname}_system_{final_it}.th" if args.compression else f"{logdir}/{args.expname}.th"
                kwargs = tensorf.get_kwargs()
                save_system_ckpt(save_path, tensorf, optimizer, aux_optimizer, final_it, kwargs)
            else:
                tensorf.save(f"{logdir}/{args.expname}.th")

    # -------------------- final evals --------------------
    if args.render_test:
        os.makedirs(f"{logdir}/imgs_test_all", exist_ok=True)
        if args.compression:
            tensorf.den_feat_codec.update(force=True)
            tensorf.app_feat_codec.update(force=True)
            tensorf.den_feat_codec.eval()
            tensorf.app_feat_codec.eval()
            if args.compress_before_volrend:
                out = tensorf.compress_with_external_codec(
                    tensorf.den_feat_codec, tensorf.app_feat_codec, mode="eval"
                )
                den_rate = sum([p["strings_length"] for p in out["den"]["rec_likelihood"]])
                app_rate = sum([p["strings_length"] for p in out["app"]["rec_likelihood"]])
                print(f"====> Final bitstream size: {(den_rate+app_rate)*1e-6:0.4f} MB")
        PSNRs_test = evaluation(
            test_dataset, tensorf, args, renderer, f"{logdir}/imgs_test_all/",
            N_vis=-1, N_samples=5, white_bg=white_bg, ndc_ray=ndc_ray, device=device,
        )
        wandb.log({"test/psnr_final": float(np.mean(PSNRs_test)), "iter": int(final_it)}, step=final_it)
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <======")

    # (optional) train-set render on pretrain path if you keep render_train=1 in your cfg
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
