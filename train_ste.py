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
from models.tensorSTE import TensorSTE, JPEGPlanesCfg


# ======================================================================================
# Globals
# ======================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast   # keep identical for compatibility


# ======================================================================================
# Utilities
# ======================================================================================

def _load_optim_states_if_any(system, optimizer, aux_optimizer):
    if ("optimizer" in system) and (system["optimizer"] is not None):
        try:
            optimizer.load_state_dict(system["optimizer"])
        except Exception as e:
            print(f"[resume] main optimizer state not loaded: {e}")
    if ("aux_optimizer" in system) and (system["aux_optimizer"] is not None) and (aux_optimizer is not None):
        try:
            aux_optimizer.load_state_dict(system["aux_optimizer"])
        except Exception as e:
            print(f"[resume] aux optimizer state not loaded: {e}")
    return system.get("global_step", 0)

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
        idx = 0 if not versions else (int(versions[-1].split("_")[-1]) + 1)
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

def _build_jpegste_cfg_from_args(args) -> JPEGPlanesCfg:
    return JPEGPlanesCfg(
        align=args.align,
        codec=args.codec_backend,                         # 'jpeg' or 'png'
        # density
        den_packing_mode=args.den_packing_mode,
        den_quant_mode=args.den_quant_mode,
        den_global_range=(args.den_global_min, args.den_global_max),
        den_quality=args.den_quality,
        den_png_level=args.den_png_level,
        den_r=args.den_r, den_c=args.den_c,
        # appearance
        app_packing_mode=args.app_packing_mode,
        app_quant_mode=args.app_quant_mode,
        app_global_range=(args.app_global_min, args.app_global_max),
        app_quality=args.app_quality,
        app_png_level=args.app_png_level,
        app_r=args.app_r, app_c=args.app_c,
    )

def _build_model(args, aabb, reso_cur, near_far):
    using_tensorste = (getattr(args, "codec_backend", "jpeg") in ("jpeg", "png"))
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
            cfg = _build_jpegste_cfg_from_args(args)
            tensorf.init_ste(cfg)
            tensorf.set_ste(bool(args.ste_enabled))
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
        cfg = _build_jpegste_cfg_from_args(args)
        tensorf.init_ste(cfg)
        tensorf.set_ste(bool(args.ste_enabled))
        tensorf.enable_vec_qat()
    else:
        raise Exception("Legacy adaptor path disabled in this script.")
    return tensorf


def _configure_optimizers(tensorf, args):
    if not args.compression:
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        return torch.optim.Adam(grad_vars, betas=(0.9, 0.99)), None

    # STE/JPEG/PNG: optimize only TensoRF (plus optional extras)
    if not args.resume_finetune:
        grad_vars = tensorf.get_optparam_groups(
            lr_init_spatialxyz=2e-3, lr_init_network=1e-4, fix_plane=args.fix_triplane
        )
    else:
        grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3, lr_init_network=0)

    if args.additional_vec:
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
        optimizer.step()

        # metrics & lr decay logging
        psnr = -10.0 * np.log(mse.detach().item()) / np.log(10.0)
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
