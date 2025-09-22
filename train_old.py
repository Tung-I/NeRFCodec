import glob
import os
import pdb
import sys
import time
import json
import random
import datetime
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict


# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast  # must remain identical for compatibility


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

class SimpleSampler:
    """Unchanged behavior: batched random permutation sampler over a fixed ray set."""

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
        return self.ids[self.curr : self.curr + self.batch]


def set_seed(seed: int):
    """Deterministic seed setup (identical to original)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
# Mesh export / Rendering entry points (unchanged external signatures & effects)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


@torch.no_grad()
def render_test(args):
    # --- Dataset init (unchanged) ---
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!!")
        return

    # --- Model init (kept behavior) ---
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    if args.rate_penalty:
        kwargs.update({"rate_penalty": True})
    if args.compression:
        kwargs.update({"compression_strategy": args.compression_strategy})

    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    if args.compression:
        if args.compression_strategy == "batchwise_img_coding":
            tensorf.init_image_codec()
        elif args.compression_strategy == "adaptor_feat_coding":
            tensorf.init_feat_codec()

    # --- Log folder layout (identical policy) ---
    logfolder = _build_log_dir(args)
    os.makedirs(logfolder)

    # === Optional plane save / reload / hist (kept disabled by default) ===
    _optional_plane_dump_and_hist(tensorf, logfolder)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


# --------------------------------------------------------------------------------------
# Training / Reconstruction
# --------------------------------------------------------------------------------------

def reconstruction(args):
    # --- Dataset init ---
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        args.datadir, split="train", downsample=args.downsample_train, is_stack=False
    )
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # --- Resolution / schedule lists ---
    upsamp_list, update_AlphaMask_list = _derive_schedule_lists(args)
    n_lamb_sigma, n_lamb_sh = args.n_lamb_sigma, args.n_lamb_sh

    # --- Logdir & writers ---
    logfolder = _build_log_dir(args)
    print(f"Please check log folder: {logfolder}")
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    with open(f"{logfolder}/train_cfg.json", "w") as json_file:
        json.dump(vars(args), json_file)

    # --- Model init (resume vs scratch), codec glue kept intact ---
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    tensorf = _build_model(args, aabb, reso_cur, near_far)

    # nSamples rule identical to original
    if args.compression:
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # Shared-MLP learning rate handling (unchanged)
    if args.shared_mlp:
        args.lr_basis = 0
        print(f"LR of mlp: {args.lr_basis}")

    # --- Optimizers (model + optional codec & aux) ---
    optimizer, aux_optimizer = _configure_optimizers_for_training(tensorf, args)

    # Feature reconstruction ref (for rec_feat_loss)
    if args.feat_rec_loss:
        tensorf.copy_pretrain_feats()

    # --- Voxel upsample schedule ---
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        )
        .long()
        .tolist()[1:]
    )

    # --- Regularizers & TV ---
    Ortho_reg_weight = args.Ortho_weight
    L1_reg_weight = args.L1_weight_inital
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()

    print("initial Ortho_reg_weight", Ortho_reg_weight)
    print("initial L1_reg_weight", L1_reg_weight)
    print(
        f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}"
    )

    # --- Learning-rate decay policy (unchanged math) ---
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    # --- Optional warm-up for codec (unchanged behavior) ---
    if args.warm_up:
        _codec_warmup_if_needed(tensorf, args, summary_writer, logfolder)

    # --- Rays, buffers, sampler ---
    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    # --- Timing accumulators ---
    forward_times, backward_times, compress_times, render_times = [], [], [], []

    # --- Training loop ---
    final_iteration = 0
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        final_iteration = iteration

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # (Optional) pre-render compression path
        start_time_forward = time.time()
        if args.compression and args.compress_before_volrend:
            if args.decode_from_latent_code:
                coding_output = tensorf.decode_all_planes()
            else:
                start_time_compress = time.time()
                coding_output = tensorf.compress_with_external_codec(
                    tensorf.den_feat_codec, tensorf.app_feat_codec
                )
                torch.cuda.synchronize()
                compress_times.append(time.time() - start_time_compress)
        else:
            coding_output = None

        # Volume rendering forward pass
        start_time_render = time.time()
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train,
            tensorf,
            chunk=32768,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True,
        )
        torch.cuda.synchronize()
        render_times.append(time.time() - start_time_render)
        del depth_map
        torch.cuda.empty_cache()

        # Rate estimation (unchanged)
        if args.compression:
            if args.compress_before_volrend:
                density_likelihood_list = coding_output["den"]["rec_likelihood"]
                app_likelihood_list = coding_output["app"]["rec_likelihood"]
            else:
                density_likelihood_list, app_likelihood_list = tensorf.get_rate()

            density_rate_loss = _rate_from_likelihoods(density_likelihood_list)
            app_rate_loss = _rate_from_likelihoods(app_likelihood_list)
            rate_loss = args.den_rate_weight * density_rate_loss + args.app_rate_weight * app_rate_loss

            summary_writer.add_scalar("train/density_rate_loss", density_rate_loss, global_step=iteration)
            summary_writer.add_scalar("train/app_rate_loss", app_rate_loss, global_step=iteration)
            summary_writer.add_scalar("train/rate_loss", rate_loss, global_step=iteration)
        else:
            rate_loss = 0.0

        # Reconstruction loss
        mse = torch.mean((rgb_map - rgb_train) ** 2)
        total_loss = mse

        # Regularization terms (identical to original)
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar("train/reg", loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar("train/reg_l1", loss_reg_L1.detach().item(), global_step=iteration)
        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss += loss_tv
            summary_writer.add_scalar("train/reg_tv_density", loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss += loss_tv
            summary_writer.add_scalar("train/reg_tv_app", loss_tv.detach().item(), global_step=iteration)
        if args.compression and args.rate_penalty:
            total_loss = total_loss + rate_loss * 1e-9
        if args.feat_rec_loss:
            feat_rec = features_rec_loss(tensorf, coding_output)
            total_loss += feat_rec * 1e-2
            summary_writer.add_scalar("train/feat_rec_loss", feat_rec, global_step=iteration)
        if args.compression:
            _clip_codec_grads_if_needed(tensorf, max_norm=1.0)
        if args.entropy_on_weight:
            entropy_loss = _entropy_on_app_weight(tensorf)
            total_loss += entropy_loss * 1e-7
            summary_writer.add_scalar(
                "train/entropy_loss_on_weight", entropy_loss * 1e-7, global_step=iteration
            )

        # Backprop on rendering loss
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time_forward)

        start_time_backward = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start_time_backward)

        # Aux loss for entropy model (compression)
        if args.compression:
            aux_optimizer.zero_grad()
            aux_loss = tensorf.get_aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            summary_writer.add_scalar("train/aux_loss", aux_loss, global_step=iteration)

        # Metrics & LR decay logging
        PSNRs.append(-10.0 * np.log(mse.detach().item()) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar("train/mse", mse.detach().item(), global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor
            summary_writer.add_scalar("lr", param_group["lr"], global_step=iteration)

        # Progress bar readout (unchanged text)
        if iteration % args.progress_refresh_rate == 0:
            desc = (
                f"Iteration {iteration:05d}:"
                f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                f" mse = {mse.detach().item():.6f}"
            )
            pbar.set_description(desc)
            PSNRs, forward_times, backward_times = [], [], []

        # Periodic visualization & compression-eval branch
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            if args.compression:
                with torch.no_grad():
                    tensorf.den_feat_codec.update(force=True)
                    tensorf.app_feat_codec.update(force=True)
                    tensorf.den_feat_codec.eval()
                    tensorf.app_feat_codec.eval()

                    if args.compress_before_volrend:
                        if args.decode_from_latent_code:
                            coding_output = tensorf.decode_all_planes(mode="eval")
                        else:
                            coding_output = tensorf.compress_with_external_codec(
                                tensorf.den_feat_codec.eval(),
                                tensorf.app_feat_codec.eval(),
                                "eval",
                            )

                    PSNRs_test = evaluation(
                        test_dataset,
                        tensorf,
                        args,
                        renderer,
                        f"{logfolder}/imgs_vis/",
                        N_vis=args.N_vis,
                        prtx=f"compress{iteration:06d}_",
                        N_samples=nSamples,
                        white_bg=white_bg,
                        ndc_ray=ndc_ray,
                        compute_extra_metrics=False,
                    )
                    summary_writer.add_scalar(
                        "test/psnr_compress", np.mean(PSNRs_test), global_step=iteration
                    )

                    tensorf.den_feat_codec.train()
                    tensorf.app_feat_codec.train()

                    den_rate = sum([item["strings_length"] for item in coding_output["den"]["rec_likelihood"]])
                    app_rate = sum([item["strings_length"] for item in coding_output["app"]["rec_likelihood"]])
                    total_mem = den_rate + app_rate
                    summary_writer.add_scalar("test/mem.", total_mem * 1e-6, global_step=iteration)

                    tensorf.save(f"{logfolder}/{args.expname}_compression_{iteration:06d}.th")
            else:
                PSNRs_test = evaluation(
                    test_dataset,
                    tensorf,
                    args,
                    renderer,
                    f"{logfolder}/imgs_vis/",
                    N_vis=args.N_vis,
                    prtx=f"{iteration:06d}_",
                    N_samples=nSamples,
                    white_bg=white_bg,
                    ndc_ray=ndc_ray,
                    compute_extra_metrics=False,
                )
                summary_writer.add_scalar("test/psnr", np.mean(PSNRs_test), global_step=iteration)

        # LR reset and AlphaMask / Upsampling schedules (identical triggers)
        if args.compression and iteration == 10000 and args.lr_reset > 0:
            grad_vars = tensorf.get_optparam_groups(2e-3, 1e-4, args.fix_triplane)
            if args.codec_training:
                if args.compression_strategy == "batchwise_img_coding":
                    codec_grad = tensorf.get_optparam_from_image_codec(1e-4, args.fix_encoder)
                else:
                    codec_grad, _ = tensorf.get_optparam_from_feat_codec(args.lr_feat_codec)
                grad_vars += codec_grad
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:
                reso_mask = reso_cur
            tensorf.alphaMask_offset = 1e-3 if (2000 < iteration < 10000) else 0
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)
            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # --- Save final checkpoints (identical filenames) ---
    if args.compression:
        tensorf.save(f"{logfolder}/{args.expname}_compression.th")
    else:
        tensorf.save(f"{logfolder}/{args.expname}.th")

    # --- Final renders / eval blocks (unchanged behavior & logging keys) ---
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset_eval = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        PSNRs_test = evaluation(
            train_dataset_eval,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================")

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        if args.compression:
            tensorf.den_feat_codec.update(force=True)
            tensorf.app_feat_codec.update(force=True)
            tensorf.den_feat_codec.eval()
            tensorf.app_feat_codec.eval()
            tensorf.mode = "eval"

            if args.compress_before_volrend:
                coding_output = tensorf.compress_with_external_codec(
                    tensorf.den_feat_codec, tensorf.app_feat_codec, "eval"
                )
                den_rate_list = coding_output["den"]["rec_likelihood"]
                app_rate_list = coding_output["app"]["rec_likelihood"]
            else:
                den_rate_list, app_rate_list = tensorf.get_rate()
            den_rate = sum([item["strings_length"] for item in den_rate_list])
            app_rate = sum([item["strings_length"] for item in app_rate_list])
            total_mem = den_rate + app_rate
            summary_writer.add_scalar("test/final mem.", total_mem, global_step=final_iteration)
            print(f"\n ======> Mem. of bitsream is {total_mem * 1e-6:0.4f} MB <========================")

        PSNRs_test = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        SummaryWriter(logfolder).add_scalar("test/psnr_all", np.mean(PSNRs_test), global_step=final_iteration)
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================")

    if args.render_path:
        c2ws = test_dataset.render_path
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


# --------------------------------------------------------------------------------------
# Helper blocks (pure refactors; preserve exact side-effects / filepaths)
# --------------------------------------------------------------------------------------

def _build_log_dir(args) -> str:
    if args.add_timestamp:
        return f"{args.basedir}/{args.expname}/{datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')}"
    if args.add_exp_version:
        exp_folder = f"{args.basedir}/{args.expname}/"
        if not os.path.exists(exp_folder):
            exp_idx = 0
        else:
            exp_dirs = sorted(glob.glob(f"{args.basedir}/{args.expname}/version_*") )
            if not exp_dirs:
                exp_idx = 0
            else:
                last_idx = int(exp_dirs[-1].split("_")[-1])
                exp_idx = last_idx + 1
        return f"{args.basedir}/{args.expname}/version_{exp_idx:03d}"
    return f"{args.basedir}/{args.expname}"


def _derive_schedule_lists(args):
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    if args.compression:
        # freeze further spatial refinement under compression mode (as in original)
        upsamp_list = [100001]
        update_AlphaMask_list = [100001]
    return upsamp_list, update_AlphaMask_list


def _build_model(args, aabb, reso_cur, near_far):
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})

        if args.compression:
            kwargs.update({"compression_strategy": args.compression_strategy})
        if args.compress_before_volrend:
            kwargs.update({"compress_before_volrend": args.compress_before_volrend})
        if args.vec_qat:
            kwargs.update({"vec_qat": args.vec_qat})
        if args.decode_from_latent_code:
            kwargs.update({"decode_from_latent_code": args.decode_from_latent_code})
        if args.compression and kwargs.get("shadingMode", args.shadingMode) != args.shadingMode:
            kwargs["shadingMode"] = args.shadingMode

        tensorf = eval(args.model_name)(**kwargs)
        if args.ckpt is not None:
            tensorf.load(ckpt)

        if args.compression:
            if args.compression_strategy == "batchwise_img_coding":
                tensorf.init_image_codec()
            elif args.compression_strategy == "adaptor_feat_coding":
                tensorf.init_feat_codec(
                    args.codec_ckpt,
                    adaptor_q_bit=args.adaptor_q_bit,
                    codec_backbone_type=args.codec_backbone_type,
                )
            else:
                raise NotImplementedError(
                    f"Not support {args.compression_strategy} for now"
                )

            if args.joint_train_from_scratch:
                tensorf.init_svd_volume(res=tensorf.gridSize, device=tensorf.device)

            if args.resume_finetune and args.system_ckpt is not None:
                system_ckpt = torch.load(args.system_ckpt, map_location=device, weights_only=False)
                # Align _quantized_cdf sizes
                app_cdf = system_ckpt["state_dict"][
                    "app_feat_codec.entropy_bottleneck._quantized_cdf"
                ].size()
                den_cdf = system_ckpt["state_dict"][
                    "den_feat_codec.entropy_bottleneck._quantized_cdf"
                ].size()
                tensorf.app_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(app_cdf)
                tensorf.den_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(den_cdf)
                tensorf.load(system_ckpt)

            if args.additional_vec:
                tensorf.init_additional_volume(device=device)
            tensorf.enable_vec_qat()

        return tensorf

    # Train from scratch path (unchanged defaults)
    tensorf = eval(args.model_name)(
        aabb,
        reso_cur,
        device,
        density_n_comp=args.n_lamb_sigma,
        appearance_n_comp=args.n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingMode,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )

    if args.shared_mlp:
        shared_module_info = torch.load("log/shared_MLP_from_scratch_exp/version_008/shared_module.pth")
        tensorf.basis_mat.load_state_dict(shared_module_info["basis_mat"])
        tensorf.renderModule.load_state_dict(shared_module_info["renderModule"])
        print("Finish loading params of share MLP")

    return tensorf


def _configure_optimizers_for_training(tensorf, args):
    # Base groups
    if not args.compression:
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        return optimizer, None

    # Compression: joint vs resume
    if not args.resume_finetune:
        grad_vars = tensorf.get_optparam_groups(
            lr_init_spatialxyz=2e-3, lr_init_network=1e-4, fix_plane=args.fix_triplane
        )
    else:
        grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3, lr_init_network=0)

    # Codec parameters
    aux_grad_vars = []
    if args.codec_training:
        if args.compression_strategy == "batchwise_img_coding":
            codec_grad_vars = tensorf.get_optparam_from_image_codec(1e-4, args.fix_encoder)
        else:
            fix_encoder_prior = hasattr(args, "fix_encoder_prior")
            codec_grad_vars, aux_grad_vars = tensorf.get_optparam_from_feat_codec(
                args.lr_feat_codec,
                fix_decoder_prior=args.fix_decoder_prior,
                fix_encoder_prior=fix_encoder_prior,
            )
        grad_vars += codec_grad_vars

    if args.additional_vec:
        grad_vars += tensorf.get_additional_optparam_groups(lr_init_spatialxyz=2e-3)
    if args.decode_from_latent_code:
        grad_vars += tensorf.get_latent_code_groups(lr_latent_code=args.lr_latent_code)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # Aux optimizer from original helper
    _, aux_optimizer = configure_optimizers(tensorf, args)
    return optimizer, aux_optimizer


def _codec_warmup_if_needed(tensorf, args, summary_writer, logfolder):
    print("warm up mode")
    if args.warm_up_ckpt == "":
        import matplotlib.pyplot as plt

        codec_grad_vars, aux_grad_vars = tensorf.get_optparam_from_feat_codec(
            args.lr_feat_codec, fix_decoder_prior=args.fix_decoder_prior, fix_encoder_prior=True
        )
        warmup_optimizer = torch.optim.Adam(codec_grad_vars, betas=(0.9, 0.99))
        os.makedirs(f"{logfolder}/warm_up_feat", exist_ok=True)

        for iter in tqdm(range(args.warm_up_iters + 1)):
            coding_output = tensorf.compress_with_external_codec(
                tensorf.den_feat_codec, tensorf.app_feat_codec, mode="train"
            )
            feat_rec_loss = features_rec_loss(tensorf, coding_output)

            warmup_optimizer.zero_grad()
            feat_rec_loss.backward()
            warmup_optimizer.step()

            summary_writer.add_scalar("warm_up/feat_rec_loss", feat_rec_loss, global_step=iter)

            if iter == args.warm_up_iters:
                torch.save(tensorf.den_feat_codec.state_dict(), f"{logfolder}/den_feat_codec_{iter}.pth")
                torch.save(tensorf.app_feat_codec.state_dict(), f"{logfolder}/app_feat_codec_{iter}.pth")

            # Optional visualization kept disabled by default for parity
        warmup_optimizer.zero_grad()
        del warmup_optimizer, coding_output, feat_rec_loss
        torch.cuda.empty_cache()
    else:
        tensorf.den_feat_codec.load_state_dict(
            torch.load(f"{args.warm_up_ckpt}/den_feat_codec_{args.warm_up_iters}.pth", weights_only=False)
        )
        tensorf.app_feat_codec.load_state_dict(
            torch.load(f"{args.warm_up_ckpt}/app_feat_codec_{args.warm_up_iters}.pth", weights_only=False)
        )
        print("Loading warm up ckpt")


def _rate_from_likelihoods(likelihood_list):
    rate = 0
    for pack in likelihood_list:
        rate += sum((torch.log(l).sum() / (-math.log(2))) for l in pack["likelihoods"].values())
    return rate


def _clip_codec_grads_if_needed(tensorf, max_norm=1.0):
    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(tensorf.den_feat_codec.parameters(), max_norm)
    torch.nn.utils.clip_grad_norm_(tensorf.app_feat_codec.parameters(), max_norm)


def _entropy_on_app_weight(tensorf):
    w_mean = tensorf.app_feat_codec.decoder_adaptor[0].weight_q.mean()
    w_std = tensorf.app_feat_codec.decoder_adaptor[0].weight_q.std()
    w_likelihoods = likelihood(tensorf.app_feat_codec.decoder_adaptor[0].weight_q, w_std, w_mean)
    return (torch.log(w_likelihoods).sum() / (-math.log(2)))


def _optional_plane_dump_and_hist(tensorf, logfolder):
    # === Saving planes as images ===
    save_planes = False
    if save_planes:
        def save_plane_to_img(plane: torch.Tensor, save_path: str):
            if plane.shape[1] < 3:
                plane = torch.cat(
                    [plane, torch.zeros([1, 3 - plane.shape[1], *plane.shape[-2:]], device=plane.device)], dim=1
                )
            norm_plane = (plane + plane.min()) / (plane.max() - plane.min())
            norm_plane_np = norm_plane[0, 0:3].permute([1, 2, 0]).detach().cpu().numpy()
            norm_plane_np = (norm_plane_np * 255).astype(np.uint8)
            imageio.imsave(save_path, norm_plane_np)

        save_dir = f"{logfolder}/planes_img"
        os.makedirs(save_dir, exist_ok=True)
        meta_path = f"{save_dir}/metadata.txt"
        if os.path.exists(meta_path):
            os.remove(meta_path)
        with open(meta_path, "a") as fp:
            for idx, set_of_planes in enumerate(tensorf.app_plane):
                group_num = (set_of_planes.shape[1] + 3 - 1) // 3
                for i in range(group_num):
                    saved_planes = set_of_planes[:, 3 * i : 3 * (i + 1)]
                    save_plane_to_img(saved_planes, os.path.join(save_dir, f"app_plane_{idx}_{i:02d}.png"))
                    fp.write(
                        f"app_plane_{idx}_{i:02d}: min:{saved_planes.min()}, max:{saved_planes.max()}\n"
                    )
            for idx, set_of_planes in enumerate(tensorf.density_plane):
                group_num = (set_of_planes.shape[1] + 3 - 1) // 3
                for i in range(group_num):
                    saved_planes = set_of_planes[:, 3 * i : 3 * (i + 1)]
                    save_plane_to_img(saved_planes, os.path.join(save_dir, f"density_plane_{idx}_{i:02d}.png"))
                    fp.write(
                        f"density_plane_{idx}_{i:02d}: min:{saved_planes.min()}, max:{saved_planes.max()}\n"
                    )

    # === Histogram (disabled by default) ===
    draw_hist = False
    if draw_hist:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        fig, ax = plt.subplots()
        num_bins = 50
        for plane_group_idx, plane in enumerate(tensorf.density_plane):
            for plane_idx in range(plane.size(1)):
                data = plane[0, plane_idx].flatten().cpu().numpy()
                weight = np.ones(len(data)) / len(data)
                cnt, bins, patches = ax.hist(data, num_bins, weights=weight, density=False)
                for i in range(len(cnt)):
                    if cnt[i] > 0.01:
                        plt.text(
                            bins[i] + (bins[i + 1] - bins[i]) / 2,
                            cnt[i],
                            f"{cnt[i]*100:.0f}",
                            ha="center",
                            va="bottom",
                        )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        fig.tight_layout()
        plt.show()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    set_seed(20211202)

    args = config_parser()
    
    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
