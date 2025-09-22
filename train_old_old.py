import glob
import os
import pdb

import torch
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import math

from dataLoader import dataset_dict
import sys
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    if args.rate_penalty:
        kwargs.update({'rate_penalty': True})
    if args.compression:
        kwargs.update({'compression_strategy': args.compression_strategy})

    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    if args.compression:
        if args.compression_strategy == "batchwise_img_coding":
            tensorf.init_image_codec()
        elif args.compression_strategy == "adaptor_feat_coding":
            tensorf.init_feat_codec()

    # logfolder = os.path.dirname(args.ckpt)

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}/{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    elif args.add_exp_version:
        exp_folder = f'{args.basedir}/{args.expname}/'
        if not os.path.exists(exp_folder):
            exp_idx = 0
        else:
            exp_dirs = sorted(glob.glob(f'{args.basedir}/{args.expname}/version_*'))
            if not exp_dirs:
                exp_idx = 0
            else:
                last_idx = int(exp_dirs[-1].split('_')[-1])
                exp_idx = last_idx + 1

        logfolder = f'{args.basedir}/{args.expname}/version_{exp_idx:03d}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    os.makedirs(logfolder)

    ### save img codec
    # tensorf.save_img_codec_ckpt(ckpt_dir=logfolder)
    ### load specific img codec
    # tensorf.load_img_codec_ckpt(logfolder, ckpt=args.codec_ckpt)


    # NeRF Codec: save 2D planes as .png files
    save_planes = False
    if save_planes:
        def save_plane_to_img(plane: torch.Tensor, save_path: str):
            '''
            plane: features on plane to be saved. Shape: [1, 3, H, W]
            save_path: path to be saved
            '''
            if plane.shape[1] < 3:
                plane = torch.cat([plane, torch.zeros([1, 3 - plane.shape[1], *plane.shape[-2:]], device=plane.device)], dim=1)

            norm_plane = (plane + plane.min()) / (plane.max() - plane.min())
            norm_plane_np = norm_plane[0, 0:3].permute(
                [1, 2, 0]).detach().cpu().numpy()  # take first 3 sub-planes from 16 planes
            norm_plane_np = (norm_plane_np * 255).astype(np.uint8)

            imageio.imsave(save_path, norm_plane_np)

        save_dir = f'{logfolder}/planes_img'
        os.makedirs(f'{logfolder}/planes_img', exist_ok=True)

        if os.path.exists(f'{save_dir}/metadata.txt'):
            os.remove(f'{save_dir}/metadata.txt')
        with open(f'{save_dir}/metadata.txt', 'a') as fp:
            # save appearance planes
            for idx, set_of_planes in enumerate(tensorf.app_plane):
                group_num = (set_of_planes.shape[1] + 3 - 1) // 3
                for i in range(group_num):
                    saved_planes = set_of_planes[:, 3 * i:3 * (i + 1), ]
                    save_plane_to_img(saved_planes,
                                      os.path.join(save_dir, f'app_plane_{idx}_{i:02d}.png'))
                    # save metadata: (min, max)
                    fp.write(f'app_plane_{idx}_{i:02d}: min:{saved_planes.min()}, max:{saved_planes.max()}' + "\n")
            # save density planes
            for idx, set_of_planes in enumerate(tensorf.density_plane):
                group_num = (set_of_planes.shape[1] + 3 - 1) // 3
                for i in range(group_num):
                    saved_planes = set_of_planes[:, 3*i:3*(i+1), ]
                    save_plane_to_img(saved_planes,
                                      os.path.join(save_dir, f'density_plane_{idx}_{i:02d}.png'))
                    # save metadata: (min, max)
                    fp.write(f'density_plane_{idx}_{i:02d}: min:{saved_planes.min()}, max:{saved_planes.max()}'+ "\n")

    # Planes Visualization
    # import matplotlib.pyplot as plt
    # # density planes
    # for plane_group_idx, plane in enumerate(tensorf.density_plane):
    #     fig_h, fig_w = 2, 8
    #     fig, axes = plt.subplots(fig_h, fig_w)
    #     for plane_idx in range(plane.size(1)):
    #         data = plane[0, plane_idx].cpu().numpy()
    #
    #         row, col = plane_idx//fig_w, plane_idx%fig_w
    #         img = axes[row][col].matshow(data, cmap='bwr', vmin=-6, vmax=6)
    #         # cbar = fig.colorbar(img, ax=axes[row][col])
    #
    #         axes[row][col].set_xticks([])
    #         axes[row][col].set_yticks([])
    #
    #         # axes[row][col].set_title(f'density_{plane_group_idx}_{plane_idx}')
    #     plt.tight_layout()
    #     plt.show()
    # # appearance planes
    # for plane_group_idx, plane in enumerate(tensorf.app_plane):
    #     fig_h, fig_w = 4, 12
    #     fig, axes = plt.subplots(fig_h, fig_w)
    #     for plane_idx in range(plane.size(1)):
    #         data = plane[0, plane_idx].cpu().numpy()
    #
    #         row, col = plane_idx//fig_w, plane_idx%fig_w
    #         img = axes[row][col].matshow(data, cmap='bwr', vmin=-10, vmax=10)
    #         # cbar = fig.colorbar(img, ax=axes[row][col])
    #
    #         axes[row][col].set_xticks([])
    #         axes[row][col].set_yticks([])
    #
    #         # axes[row][col].set_title(f'density_{plane_group_idx}_{plane_idx}')
    #     plt.tight_layout()
    #     plt.show()


    ### Hist of planes
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots()
    num_bins = 50

    draw_hist = False
    if draw_hist:
        # density planes
        for plane_group_idx, plane in enumerate(tensorf.density_plane):
            for plane_idx in range(plane.size(1)):
                data = plane[0, plane_idx].flatten().cpu().numpy()
                weight = np.ones(len(data)) / len(data)
                cnt, bins, patches = ax.hist(data, num_bins, weights=weight, density=False)
                for i in range(len(cnt)):
                    if cnt[i] > 0.01: # ratio > 1%
                        plt.text(bins[i] + (bins[i + 1] - bins[i]) / 2, cnt[i], f'{cnt[i]*100:.0f}', ha='center', va='bottom')
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            fig.tight_layout()
            plt.show()


    reload_planes = False
    # assert reload_planes != save_planes
    def read_image(filepath: str) -> torch.Tensor:
        from torchvision import transforms
        # assert filepath.is_file()
        img = Image.open(filepath).convert("RGB")
        return transforms.ToTensor()(img)

    def parse_meta(meta_log: str):
        plane_id = meta_log.split(',')[0].split(':')[0]
        min = meta_log.split(',')[0].split(':')[-1]
        max = meta_log.split(',')[1].split(':')[-1]

        return plane_id, float(min), float(max)

    if reload_planes:
        planes_dict = {}
        planes_list = []
        rec_img_files = sorted(glob.glob(f'./{logfolder}/planes_img/*-ans.png'))

        with open(f'./{logfolder}/planes_img/metadata.txt', "r") as meta:
            for i in range(len(rec_img_files)):
                meta_log = meta.readline()
                print(meta_log)
                plane_id, min, max = parse_meta(meta_log)

                plane = read_image(rec_img_files[i])
                plane = min + (max - min) * plane

                planes_list.append(plane)
                # planes_dict.update({plane_id: plane})

        for idx, set_of_planes in enumerate(tensorf.app_plane):
            group_num = (set_of_planes.shape[1] + 3 - 1) // 3
            for i in range(group_num):
                set_of_planes[:, 3*i:3*(i+1)] = planes_list[i+idx*group_num].unsqueeze(dim=0)


        for idx, set_of_planes in enumerate(tensorf.density_plane):
            total_planes = set_of_planes.shape[1]
            group_num = (set_of_planes.shape[1] + 3 - 1) // 3
            for i in range(group_num):
                if i == (group_num - 1):
                    set_of_planes[:, 3*i:] = planes_list[i+idx*group_num+tensorf.app_plane[0].shape[1]][0:(total_planes-3*i)].unsqueeze(dim=0)
                else:
                    set_of_planes[:, 3*i:3*(i+1)] = planes_list[i+idx*group_num+tensorf.app_plane[0].shape[1]].unsqueeze(dim=0)

    # storage
    if False:
        import json
        storage = 0
        results_files_list = sorted(glob.glob(f'./{logfolder}/planes_img/*.json'))
        results_files_list = [item for item in results_files_list if os.path.basename(item)[0:3] == 'app' or os.path.basename(item)[0:3] == 'den']
        pdb.set_trace()
        for idx, set_of_planes in enumerate(tensorf.app_plane):
            group_num = (set_of_planes.shape[1] + 3 - 1) // 3
            for i in range(group_num):
                with open(results_files_list[i+idx*group_num], "r") as json_file:
                    data = json.load(json_file)
                    storage += data["results"]["bpp"] * set_of_planes.shape[-2] * set_of_planes.shape[-1] / 8.

        print(f"app {storage * 1e-6} MB")
        pdb.set_trace()
        for idx, set_of_planes in enumerate(tensorf.density_plane):
            group_num = (set_of_planes.shape[1] + 3 - 1) // 3
            for i in range(group_num):
                with open(results_files_list[i+idx*group_num+tensorf.app_plane[0].shape[1]], "r") as json_file:
                    data = json.load(json_file)
                    storage += data["results"]["bpp"] * set_of_planes.shape[-2] * set_of_planes.shape[-1] / 8.

        print(f"total {storage * 1e-6} MB")


    # pdb.set_trace()
    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    #     PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    #     print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')
    #
    # if args.render_test:
    #     os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
    #     evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    #     print(f"saved @ {logfolder}/{args.expname}/imgs_test_all ")

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    if args.compression:
        upsamp_list = [100001]
        update_AlphaMask_list = [100001]
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh


    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}/{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    elif args.add_exp_version:
        exp_folder = f'{args.basedir}/{args.expname}/'
        if not os.path.exists(exp_folder):
            exp_idx = 0
        else:
            exp_dirs = sorted(glob.glob(f'{args.basedir}/{args.expname}/version_*'))
            if not exp_dirs:
                exp_idx = 0
            else:
                last_idx = int(exp_dirs[-1].split('_')[-1])
                exp_idx = last_idx + 1

        logfolder = f'{args.basedir}/{args.expname}/version_{exp_idx:03d}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    print(f"Please check log folder: {logfolder}")


    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    cfg_dict = vars(args)
    with open(f"{logfolder}/train_cfg.json", 'w') as json_file:
        json.dump(cfg_dict, json_file)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})

        # pass arguments into model
        if args.compression:
            kwargs.update({'compression_strategy': args.compression_strategy})
        if args.compress_before_volrend:
            kwargs.update({'compress_before_volrend': args.compress_before_volrend})
        if args.vec_qat:
            kwargs.update({'vec_qat': args.vec_qat})
        if args.decode_from_latent_code:
            kwargs.update({'decode_from_latent_code': args.decode_from_latent_code})
        # if args.compression and args.rate_penalty:
        #     kwargs.update({'rate_penalty': True})

        if args.compression:
            if kwargs['shadingMode'] != args.shadingMode:
                kwargs['shadingMode'] = args.shadingMode

        # 这样写是为了 当 目前没有codec时，不会报错
        if not args.compression:
            tensorf = eval(args.model_name)(**kwargs)
            tensorf.load(ckpt)

        else:
            tensorf = eval(args.model_name)(**kwargs)
            if args.ckpt is not None:
                tensorf.load(ckpt)
            if args.compression_strategy == 'batchwise_img_coding':
                tensorf.init_image_codec()
            elif args.compression_strategy == 'adaptor_feat_coding':
                tensorf.init_feat_codec(args.codec_ckpt, adaptor_q_bit=args.adaptor_q_bit,
                                        codec_backbone_type=args.codec_backbone_type)
            else:
                raise NotImplementedError(f"Not support {args.compression_strategy} for now")

            # TODO: add joint training from scratch
            if args.joint_train_from_scratch:
                tensorf.init_svd_volume(res=tensorf.gridSize, device=tensorf.device)

            # load trained TensoRF + codec ckpt
            if args.resume_finetune and args.system_ckpt is not None:
                system_ckpt = torch.load(args.system_ckpt, map_location=device)
                # alignment of "_quantized_cdf"
                app_feat_quantized_cdf = system_ckpt["state_dict"][
                    "app_feat_codec.entropy_bottleneck._quantized_cdf"].size()
                den_feat_quantized_cdf = system_ckpt["state_dict"][
                    "den_feat_codec.entropy_bottleneck._quantized_cdf"].size()
                tensorf.app_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(app_feat_quantized_cdf)
                tensorf.den_feat_codec.entropy_bottleneck._quantized_cdf = torch.zeros(den_feat_quantized_cdf)

                tensorf.load(system_ckpt)

            if args.additional_vec:
                tensorf.init_additional_volume(device=device)

            # if args.vec_qat:
            tensorf.enable_vec_qat()

    else: # train from scratch
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    )

        # load share mlp
        if args.shared_mlp:
            shared_module_info = torch.load("log/shared_MLP_from_scratch_exp/version_008/shared_module.pth")
            tensorf.basis_mat.load_state_dict(shared_module_info["basis_mat"])
            tensorf.renderModule.load_state_dict(shared_module_info["renderModule"])
            print(f"Finish loading params of share MLP")

    if args.compression:
        nSamples = min(args.nSamples, tensorf.nSamples)
    else:
        nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if args.shared_mlp:
        args.lr_basis = 0
        print(f"LR of mlp: {args.lr_basis}")

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis) # 2e-2, 1e-3
    # 如果不是从头开始训练，lr都需要调小一点
    if args.compression:
        if not args.resume_finetune: # codec + fields joint training
            grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3,
                                                    lr_init_network=1e-4,
                                                    fix_plane=args.fix_triplane) # 1e-3 , 1e-4 better
        else: # resume training
            grad_vars = tensorf.get_optparam_groups(lr_init_spatialxyz=2e-3, lr_init_network=0)

        if args.codec_training:

            if args.compression_strategy == 'batchwise_img_coding':
                codec_grad_vars = tensorf.get_optparam_from_image_codec(1e-4, args.fix_encoder)
            else:

                fix_encoder_prior = hasattr(args, "fix_encoder_prior")
                codec_grad_vars, aux_grad_vars = \
                    tensorf.get_optparam_from_feat_codec(args.lr_feat_codec,
                                                        fix_decoder_prior=args.fix_decoder_prior,
                                                        fix_encoder_prior=fix_encoder_prior)
            grad_vars += codec_grad_vars

        if args.additional_vec:
            grad_vars += tensorf.get_additional_optparam_groups(lr_init_spatialxyz=2e-3)

        if args.decode_from_latent_code:
            grad_vars += tensorf.get_latent_code_groups(lr_latent_code=args.lr_latent_code)

    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    # optimizer for TensoRF & Codec
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    # aux_optimizer = torch.optim.Adam(aux_grad_vars, betas=(0.9, 0.99))
    if args.compression:
        _, aux_optimizer = configure_optimizers(tensorf, args) # appearance & density codec是否需要分开？

    ### record pretrained triplane
    # necessary for rec_feat_loss
    if args.feat_rec_loss:
        tensorf.copy_pretrain_feats()


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    ### eval before train
    # tensorf.set_external_codec_flag()
    # coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec,
    #                                                      tensorf.app_feat_codec,
    #                                                      mode="train")
    # PSNRs_test = evaluation(test_dataset, tensorf, None, renderer, f'{logfolder}/eval_before_train/', N_vis=5,
    #                         prtx=f'{0:06d}_', N_samples=tensorf.nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
    #                         compute_extra_metrics=False)
    #
    # print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    # warm up
    # via feat plane rec. loss
    if args.warm_up:
        print("warm up mode")
        if args.warm_up_ckpt == "":
            import matplotlib.pyplot as plt
            # set up optim. for codec
            codec_grad_vars, aux_grad_vars = tensorf.get_optparam_from_feat_codec(args.lr_feat_codec,
                                                                                  fix_decoder_prior=args.fix_decoder_prior,
                                                                                  fix_encoder_prior=True)
            warmup_optimizer = torch.optim.Adam(codec_grad_vars, betas=(0.9, 0.99))
            os.makedirs(f'{logfolder}/warm_up_feat', exist_ok=True)

            for iter in tqdm(range(args.warm_up_iters+1)):
                coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec,
                                                                     tensorf.app_feat_codec,
                                                                     mode="train")
                feat_rec_loss = features_rec_loss(tensorf, coding_output)

                warmup_optimizer.zero_grad()
                feat_rec_loss.backward()
                warmup_optimizer.step()

                # log mse of feat plane
                summary_writer.add_scalar('warm_up/feat_rec_loss', feat_rec_loss, global_step=iter)

                if iter == args.warm_up_iters:
                    torch.save(tensorf.den_feat_codec.state_dict(), f"{logfolder}/den_feat_codec_{iter}.pth")
                    torch.save(tensorf.app_feat_codec.state_dict(), f"{logfolder}/app_feat_codec_{iter}.pth")

                feat_vis = False
                if iter % 10000 == 9999 and feat_vis:
                    # save feat img for vis.
                    fig_h, fig_w = 2, 2
                    fig, axes = plt.subplots(fig_h, fig_w)

                    input_plane = tensorf.density_plane[0][0,0].detach().cpu().numpy()
                    rec_plane = coding_output["den"]["rec_planes"][0][0,0].detach().cpu().numpy()

                    axes[0][0].matshow(input_plane, cmap='bwr', vmin=-6, vmax=6)
                    axes[0][1].matshow(rec_plane,   cmap='bwr', vmin=-6, vmax=6)

                    input_plane = tensorf.app_plane[0][0,0].detach().cpu().numpy()
                    rec_plane = coding_output["app"]["rec_planes"][0][0,0].detach().cpu().numpy()

                    axes[1][0].matshow(input_plane, cmap='bwr', vmin=-6, vmax=6)
                    axes[1][1].matshow(rec_plane,   cmap='bwr', vmin=-6, vmax=6)

                    # clear ticks
                    for i in range(2):
                        for j in range(2):
                            axes[i][j].set_xticks([])
                            axes[i][j].set_yticks([])

                    plt.tight_layout()
                    plt.savefig(f'{logfolder}/warm_up_feat/{iter:05d}.png')
                    plt.close()

            warmup_optimizer.zero_grad()
            del warmup_optimizer
            del coding_output
            del feat_rec_loss
            torch.cuda.empty_cache()
        else:
            #load
            tensorf.den_feat_codec.load_state_dict(torch.load(f"{args.warm_up_ckpt}/den_feat_codec_{args.warm_up_iters}.pth"))
            tensorf.app_feat_codec.load_state_dict(torch.load(f"{args.warm_up_ckpt}/app_feat_codec_{args.warm_up_iters}.pth"))
            print("Loading warm up ckpt")
            


    forward_times = []
    backward_times = []

    compress_times = []
    render_times = []

    print(f"nSamples:{nSamples}")
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # if args.compression:
        #     nSamples = tensorf.nSamples

        start_time_forward = time.time()
        if args.compression and args.compress_before_volrend: # compress part
            if args.decode_from_latent_code: # auto-decoding
                coding_output = tensorf.decode_all_planes()
            else: # auto-encoding
                start_time_compress = time.time()

                coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec, tensorf.app_feat_codec)

                torch.cuda.synchronize()
                compress_time = time.time() - start_time_compress
                compress_times.append(compress_time)

        start_time_render = time.time()

        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=32768, # args.batch_size
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
        torch.cuda.synchronize()
        render_time = time.time() - start_time_render
        render_times.append(render_time)

        del depth_map
        torch.cuda.empty_cache()


        if args.compression:
            ### rate estimation:
            if args.compress_before_volrend:
                density_likelihood_list = coding_output['den']['rec_likelihood']
                app_likelihood_list = coding_output['app']['rec_likelihood']
            else:
                density_likelihood_list, app_likelihood_list = tensorf.get_rate()
            density_rate_loss = 0
            app_rate_loss = 0
            for idx, density_likelihood in enumerate(density_likelihood_list):
                rate = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2)))
                    for likelihoods in density_likelihood["likelihoods"].values()
                )
                density_rate_loss += rate

            for idx, app_likelihood in enumerate(app_likelihood_list):
                rate = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2)))
                    for likelihoods in app_likelihood["likelihoods"].values()
                )
                app_rate_loss += rate

            rate_loss = args.den_rate_weight * density_rate_loss + \
                         args.app_rate_weight * app_rate_loss
            # TODO: add aux loss, which is crucial for ANS compression

            summary_writer.add_scalar('train/density_rate_loss', density_rate_loss, global_step=iteration)
            summary_writer.add_scalar('train/app_rate_loss', app_rate_loss, global_step=iteration)
            summary_writer.add_scalar('train/rate_loss', rate_loss, global_step=iteration)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss

        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
        # rate loss
        if args.compression and args.rate_penalty:
            total_loss = total_loss + rate_loss * 1e-9 #cheng2020 1e-6/1e-7 # hyper 1e-9

        if args.feat_rec_loss: #  and iteration > 5000
            feat_rec_loss = features_rec_loss(tensorf, coding_output)

            total_loss += feat_rec_loss * 1e-2

            summary_writer.add_scalar('train/feat_rec_loss', feat_rec_loss, global_step=iteration)

        if args.compression:
            clip_max_norm = 1.0
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(tensorf.den_feat_codec.parameters(), clip_max_norm)
                torch.nn.utils.clip_grad_norm_(tensorf.app_feat_codec.parameters(), clip_max_norm)

        if args.entropy_on_weight:
            w_mean = tensorf.app_feat_codec.decoder_adaptor[0].weight_q.mean()
            w_std = tensorf.app_feat_codec.decoder_adaptor[0].weight_q.std()

            w_likelihoods = likelihood(tensorf.app_feat_codec.decoder_adaptor[0].weight_q, w_std, w_mean)
            entropy_on_app_weight = (torch.log(w_likelihoods).sum() / (-math.log(2)))

            total_loss += entropy_on_app_weight * 1e-7
            summary_writer.add_scalar('train/entropy_loss_on_weight', entropy_on_app_weight * 1e-7, global_step=iteration)

        torch.cuda.synchronize()
        forward_time = time.time() - start_time_forward
        forward_times.append(forward_time)

        start_time_backward = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time_backward
        backward_times.append(backward_time)

        # if args.compression and iteration % 500 == 0:
        #     if type(tensorf.app_feat_codec.decoder_adaptor) == nn.Sequential:
        #         summary_writer.add_histogram('dec_adaptor/app_weight', tensorf.app_feat_codec.decoder_adaptor[0].weight_q)
        #         summary_writer.add_histogram('dec_adaptor/den_weight', tensorf.den_feat_codec.decoder_adaptor[0].weight_q)
        #
        #         summary_writer.add_histogram('dec_adaptor/app_bias', tensorf.app_feat_codec.decoder_adaptor[0].bias)
        #         summary_writer.add_histogram('dec_adaptor/den_bias', tensorf.den_feat_codec.decoder_adaptor[0].bias)
        #     else:
        #         summary_writer.add_histogram('dec_adaptor/app_weight', tensorf.app_feat_codec.decoder_adaptor.weight)
        #         summary_writer.add_histogram('dec_adaptor/den_weight', tensorf.den_feat_codec.decoder_adaptor.weight)
        #
        #         summary_writer.add_histogram('dec_adaptor/app_bias', tensorf.app_feat_codec.decoder_adaptor.bias)
        #         summary_writer.add_histogram('dec_adaptor/den_bias', tensorf.den_feat_codec.decoder_adaptor.bias)

        if args.compression:
            ### aux loss term:
            aux_optimizer.zero_grad() # missed in previous version
            aux_loss = tensorf.get_aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            summary_writer.add_scalar('train/aux_loss', aux_loss, global_step=iteration)

        mse = torch.mean((rgb_map - rgb_train) ** 2).detach().item()
        
        PSNRs.append(-10.0 * np.log(mse) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', mse, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
            summary_writer.add_scalar('lr', param_group['lr'], global_step=iteration)

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            description = f'Iteration {iteration:05d}:' \
                          + f' train_psnr = {float(np.mean(PSNRs)):.2f}' \
                          + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}' \
            + f' mse = {mse:.6f}'
                # + f' forward:{float(np.mean(forward_times)):.3f} s' \
                # + f' backward:{float(np.mean(backward_times)):.3f} s' \
                # + f' compress:{float(np.mean(compress_times)):.3f} s' \
                # + f' render:{float(np.mean(render_times)):.3f} s' \

            # if args.compression:
            #     description += f' aux = {aux_loss}'

            pbar.set_description(description)
            PSNRs = []
            forward_times = []
            backward_times = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
        # if iteration % args.vis_every == 0 and args.N_vis != 0:

            # tensorf.mode = "eval"

            # forward()
            # if args.compression and args.compress_before_volrend: # compress part
            #     coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec,
            #                                                          tensorf.app_feat_codec,
            #                                                          "train") # "train/eval"
            #
            # PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
            #                         prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            #
            # summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

            # compress()
            if args.compression:
                with torch.no_grad():
                    tensorf.den_feat_codec.update(force=True)
                    tensorf.app_feat_codec.update(force=True)
                    tensorf.den_feat_codec.eval()
                    tensorf.app_feat_codec.eval()
                    if args.compression and args.compress_before_volrend:  # compress part
                        if args.decode_from_latent_code:
                            coding_output = tensorf.decode_all_planes(mode="eval")
                        else:
                            coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec.eval(),
                                                                                 tensorf.app_feat_codec.eval(),
                                                                                 "eval")  # "train/eval"

                    PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                            prtx=f'compress{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
                                            compute_extra_metrics=False)

                    summary_writer.add_scalar('test/psnr_compress', np.mean(PSNRs_test), global_step=iteration)

                    tensorf.den_feat_codec.train()
                    tensorf.app_feat_codec.train()
                    # 统计bitstream
                    den_rate_list = coding_output['den']['rec_likelihood']
                    app_rate_list = coding_output['app']['rec_likelihood']
                    den_rate = sum([item['strings_length'] for item in den_rate_list])
                    app_rate = sum([item['strings_length'] for item in app_rate_list])
                    total_mem = den_rate + app_rate
                    summary_writer.add_scalar('test/mem.', total_mem * 1e-6, global_step=iteration)

                    tensorf.save(f'{logfolder}/{args.expname}_compression_{iteration:06d}.th')
            else:
                PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)

                summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if args.compression:
            if iteration==10000 and args.lr_reset > 0:
                grad_vars = tensorf.get_optparam_groups(2e-3, 1e-4, args.fix_triplane)  # 1e-3 , 1e-4 better
                if args.codec_training:
                    if args.compression_strategy == 'batchwise_img_coding':
                        codec_grad_vars = tensorf.get_optparam_from_image_codec(1e-4, args.fix_encoder)
                    else:
                        codec_grad_vars, aux_grad_vars = tensorf.get_optparam_from_feat_codec(args.lr_feat_codec)  # best from now: 2e-4
                    grad_vars += codec_grad_vars

                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur

            if 2000 < iteration < 10000:
                tensorf.alphaMask_offset = 1e-3 # default 1e-4
            else:
                tensorf.alphaMask_offset = 0

            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))

            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
    if args.compression:
        tensorf.save(f'{logfolder}/{args.expname}_compression.th')
    else:
        tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        if args.compression: # compression mode
            tensorf.den_feat_codec.update(force=True)
            tensorf.app_feat_codec.update(force=True)
            tensorf.den_feat_codec.eval()
            tensorf.app_feat_codec.eval()
            tensorf.mode = "eval"

            if args.compress_before_volrend:
                # add compress op.
                if args.compression and args.compress_before_volrend:  # compress part
                    coding_output = tensorf.compress_with_external_codec(tensorf.den_feat_codec,
                                                                         tensorf.app_feat_codec,
                                                                         "eval")

                den_rate_list = coding_output['den']['rec_likelihood']
                app_rate_list = coding_output['app']['rec_likelihood']
            else:
                den_rate_list, app_rate_list = tensorf.get_rate()
            den_rate = sum([item['strings_length'] for item in den_rate_list])
            app_rate = sum([item['strings_length'] for item in app_rate_list])
            total_mem = den_rate + app_rate
            summary_writer.add_scalar('test/final mem.', total_mem, global_step=iteration)
            print(f"\n ======> Mem. of bitsream is {total_mem * 1e-6:0.4f} MB <========================")


        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>', c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.



if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    # torch.manual_seed(20211202)
    # np.random.seed(20211202)
    set_seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

