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

def _maybe_align_cdf_and_load_system(tensorf, system_ckpt_path):
    """Load a 'system' checkpoint (full model incl. codec), aligning entropy CDF shapes if needed."""
    system = torch.load(system_ckpt_path, map_location=device, weights_only=False)

    # Align CDF buffers before load (prevents size mismatch across compressai versions)
    def _maybe_zero_like(dest_mod, key):
        if key in system["state_dict"]:
            src_sz = system["state_dict"][key].size()
            try:
                getattr_from = dict(tensorf.named_buffers())
                cur = getattr_from[key]
                if cur.size() != src_sz:
                    # replace with zero buffer of src size so .load_state_dict works
                    parts = key.split(".")
                    mod = tensorf
                    for p in parts[:-1]:
                        mod = getattr(mod, p)
                    setattr(mod, parts[-1], torch.zeros(src_sz, device=cur.device, dtype=cur.dtype))
            except Exception:
                pass

    _maybe_zero_like(tensorf, "app_feat_codec.entropy_bottleneck._quantized_cdf")
    _maybe_zero_like(tensorf, "den_feat_codec.entropy_bottleneck._quantized_cdf")

    # Load everything (TensoRF + codec)
    tensorf.load(system, strict=False)
    return system


def _load_optim_states_if_any(system, optimizer, aux_optimizer):
    """Restore optimizer states if present."""
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
    """Mirror tensorf.save(): pack model kwargs, state_dict, and alphaMask bundle."""
    kwargs = tensorf.get_kwargs()
    ckpt = {"kwargs": kwargs, "state_dict": tensorf.state_dict()}

    if getattr(tensorf, "alphaMask", None) is not None:
        alpha_volume = tensorf.alphaMask.alpha_volume.bool().cpu().numpy()
        ckpt["alphaMask.shape"] = alpha_volume.shape
        ckpt["alphaMask.mask"]  = np.packbits(alpha_volume.reshape(-1))
        ckpt["alphaMask.aabb"]  = tensorf.alphaMask.aabb.cpu()
    return ckpt

def _save_system_ckpt(path, tensorf, optimizer, aux_optimizer, global_step, kwargs_override=None):
    """
    Save a resume-able checkpoint that still contains the same fields
    tensorf.save() used to write (incl. alphaMask) PLUS optimizers & step.
    """
    base = _make_model_ckpt_dict(tensorf)

    # Optionally override kwargs (e.g., when saving right after building from args)
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


def _codec_warmup_if_needed(tensorf, args, writer, logdir):
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
        writer.add_scalar("warm_up/feat_rec_loss", loss, global_step=it)

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
    Two backends:
      - 'jpeg'    : TensorSTE (STE+JPEG planes)
      - 'adaptor' : TensorVMSplit + CompressAI adaptors (legacy)
    """
    using_jpeg = (getattr(args, "codec_backend", "jpeg") == "jpeg")
    Model = eval(args.model_name) if hasattr(args, "model_name") else (TensorSTE if using_jpeg else TensorVMSplit)

    # --------- resume path (pretrained TensoRF -> finetune) ----------
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        kwargs = ckpt["kwargs"]; kwargs.update({"device": device})

        # align finetune flags
        if args.compression:
            kwargs.update({"compress_before_volrend": True})
            if kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
                kwargs["shadingMode"] = args.shadingMode

        tensorf = Model(**kwargs)
        tensorf.load(ckpt)

        if using_jpeg:
            # init JPEG config
            jpeg_cfg = JPEGPlanesCfg(
                plane_packing_mode=args.jpeg_plane_packing_mode,
                quant_mode=args.jpeg_quant_mode,
                global_range=(args.jpeg_global_min, args.jpeg_global_max),
                align=args.jpeg_align,
                quality=args.jpeg_quality,
            )
            tensorf.init_ste(jpeg_cfg)
            tensorf.set_ste(bool(args.ste_enabled))
            tensorf.enable_vec_qat()  # optional: QAT for line factors only
        else:
            raise Exception("Adaptor codec finetune from TensoRF not supported; please use JPEG+STE backend.")

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

    if using_jpeg:
        jpeg_cfg = JPEGPlanesCfg(
            plane_packing_mode=args.jpeg_plane_packing_mode,
            quant_mode=args.jpeg_quant_mode,
            global_range=(args.jpeg_global_min, args.jpeg_global_max),
            align=args.jpeg_align,
            quality=args.jpeg_quality,
        )
        tensorf.init_ste(jpeg_cfg)
        tensorf.set_ste(bool(args.ste_enabled))
        tensorf.enable_vec_qat()
    else:
        raise Exception("Adaptor codec training from scratch not supported; please use JPEG+STE backend.")

    return tensorf



def _configure_optimizers(tensorf, args):
    using_jpeg = (getattr(args, "codec_backend", "jpeg") == "jpeg")

    if not args.compression:
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        return torch.optim.Adam(grad_vars, betas=(0.9, 0.99)), None

    if using_jpeg:
        # Optimize only TensoRF (and optional extras); no aux optimizer
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

    # ---- legacy adaptor backend (unchanged) ----
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
    _, aux_opt = configure_optimizers(tensorf, args)
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
    writer = SummaryWriter(logdir)
    with open(f"{logdir}/train_cfg.json", "w") as f:
        json.dump(vars(args), f)

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

    start_iter = 0
    if getattr(args, "resume_system_ckpt", None):
        # load optimizer states (optional)
        if getattr(args, "resume_optim", 0):
            system = torch.load(args.resume_system_ckpt, map_location=device, weights_only=False)
            start_iter = system.get("global_step", 0)
            _load_optim_states_if_any(system, optimizer, aux_optimizer)

    # how many more steps to run
    extra_iters = getattr(args, "extra_iters", 0) or args.n_iters
    end_iter = start_iter + extra_iters
    print(f"[resume] starting at iter={start_iter}, running to {end_iter}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for it in pbar:
        final_it = it
        ray_idx = sampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # pre-render compression (adaptor path) if requested
        using_jpeg = (getattr(args, "codec_backend", "jpeg") == "jpeg")

        if args.compression and args.compress_before_volrend:
            if using_jpeg:
                coding_output = tensorf.compress_with_external_codec(mode="train")
            else:
                raise Exception("Adaptor codec not supported; please use JPEG+STE backend.")
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
            if using_jpeg:
                # JPEG path: no entropy likelihoods; optionally *log* bits but don't penalize.
                den_bits = sum(p["bits"] for p in coding_output["den"]["rec_likelihood"])
                app_bits = sum(p["bits"] for p in coding_output["app"]["rec_likelihood"])
                writer.add_scalar("train/den_bits", den_bits, global_step=it)
                writer.add_scalar("train/app_bits", app_bits, global_step=it)
                rate_loss = 0.0   # or keep a tiny penalty if you want: args.rate_weight * (den_bits+app_bits)
            else:
                raise Exception("Adaptor codec not supported; please use JPEG+STE backend.")
        else:
            rate_loss = 0.0

        # reconstruction loss + regs
        mse = torch.mean((rgb_map - rgb_train) ** 2)
        loss = mse

        if Ortho_w > 0:
            loss_reg = tensorf.vector_comp_diffs()
            loss += Ortho_w * loss_reg
            writer.add_scalar("train/reg", loss_reg.detach().item(), global_step=it)
        if L1_w > 0:
            loss_l1 = tensorf.density_L1()
            loss += L1_w * loss_l1
            writer.add_scalar("train/reg_l1", loss_l1.detach().item(), global_step=it)
        if TV_w_d > 0:
            TV_w_d *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_w_d
            loss += loss_tv
            writer.add_scalar("train/reg_tv_density", loss_tv.detach().item(), global_step=it)
        if TV_w_a > 0:
            TV_w_a *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_w_a
            loss += loss_tv
            writer.add_scalar("train/reg_tv_app", loss_tv.detach().item(), global_step=it)
        if args.compression and args.rate_penalty:
            loss = loss + rate_loss * 1e-9
        if getattr(args, "feat_rec_loss", 0):
            feat_rec = features_rec_loss(tensorf, coding_output)
            loss += 1e-2 * feat_rec
            writer.add_scalar("train/feat_rec_loss", feat_rec, global_step=it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # metrics & lr decay logging
        PSNRs.append(-10.0 * np.log(mse.detach().item()) / np.log(10.0))
        writer.add_scalar("train/PSNR", PSNRs[-1], global_step=it)
        writer.add_scalar("train/mse", mse.detach().item(), global_step=it)
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * lr_factor
            writer.add_scalar("lr", g["lr"], global_step=it)

        if it % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"it {it:05d} | train_psnr {np.mean(PSNRs):.2f} | test_psnr {np.mean(PSNRs_test):.2f} | mse {mse:.6f}"
            )
            PSNRs = []

        # periodic visualization on test set
        if it % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            if args.compression:
                if using_jpeg:
                    with torch.no_grad():
                        _ = tensorf.compress_with_external_codec(mode="eval")
                        PSNRs_test = evaluation(
                            test_dataset, tensorf, args, renderer, f"{logdir}/imgs_vis/",
                            N_vis=args.N_vis, prtx=f"jpeg{it:06d}_", N_samples=nSamples,
                            white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                        )
                        writer.add_scalar("test/psnr_compress", np.mean(PSNRs_test), global_step=it)
                else:
                    raise Exception("Adaptor codec not supported; please use JPEG+STE backend.")

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
                _save_system_ckpt(save_path, tensorf, optimizer, aux_optimizer, final_it, kwargs)
            else:
                tensorf.save(f"{logdir}/{args.expname}.th")

    # -------------------- final evals --------------------
    if args.render_test:
        os.makedirs(f"{logdir}/imgs_test_all", exist_ok=True)
        if args.compression:
            if using_jpeg:
                out = tensorf.compress_with_external_codec(mode="eval")
                den_bits = sum([p["bits"] for p in out["den"]["rec_likelihood"]])
                app_bits = sum([p["bits"] for p in out["app"]["rec_likelihood"]])
                print(f"====> Final JPEG size: {(den_bits+app_bits)*1e-6:0.4f} Mb (bits)")
            else:
               raise Exception("Adaptor codec not supported; please use JPEG+STE backend.")
        PSNRs_test = evaluation(
            test_dataset, tensorf, args, renderer, f"{logdir}/imgs_test_all/",
            N_vis=-1, N_samples=10, white_bg=white_bg, ndc_ray=ndc_ray, device=device,
        )
        SummaryWriter(logdir).add_scalar("test/psnr_all", np.mean(PSNRs_test), global_step=final_it)
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
