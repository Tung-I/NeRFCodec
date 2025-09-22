import pdb

import torch,os,imageio,sys,copy
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
# from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
# from models.triplane import TriPlane
from models.tensoRF import TensorVMSplit
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda',
                                plane_feature=None, line_feature=None, alpha_mask=None):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]

    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):

        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        if plane_feature is None:
            rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        else:
            rgb_map, depth_map = tensorf.forward_with_feature(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                                              N_samples=N_samples, plane_feature=plane_feature, line_feature=line_feature,
                                                              alphaMask=alpha_mask)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        del rgb_map
        del depth_map

    # pdb.set_trace()
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def Multiple_scene_renderer(batch_data, tensorf, chunk=4096, N_samples=-1, ndc_ray=False,
                            white_bg=True, is_train=False, device='cuda',
                            den_plane_feature=None, app_plane_feature=None,):
    B = len(batch_data)
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = batch_data[0]["rays"].shape[0] * B

    iter_for_single_scene = batch_data[0]["rays"].shape[0] // chunk

    batch_rays = torch.cat([batch_data[i]["rays"] for i in range(B)])
    batch_line_feature = [
        {
        "den": batch_data[b_idx]["model"].density_line,
        "app": batch_data[b_idx]["model"].app_line,
    } for b_idx in range(B)
    ]
    batch_alpha_mask = [batch_data[b_idx]["model"].alphaMask for b_idx in range(B)]
    batch_kwargs = []
    for b_idx in range(B):
        kwargs = batch_data[b_idx]["model"].get_kwargs()
        kwargs.update({'device': device})
        batch_kwargs += [kwargs]

    # debug
    # shared_basis_mat = copy.deepcopy(batch_data[0]["model"].basis_mat.state_dict())
    # shared_MLP = copy.deepcopy(batch_data[0]["model"].renderModule.state_dict())
    # tensorf.basis_mat.load_state_dict(shared_basis_mat)
    # tensorf.renderModule.load_state_dict(shared_MLP)


    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):

        rays_chunk = batch_rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        b_idx = chunk_idx // iter_for_single_scene
        tensorf.reload_kwargs(**batch_kwargs[b_idx])

        plane_feat_dict = {
            "den": den_plane_feature[b_idx],
            "app": app_plane_feature[b_idx],
        }

        rgb_map, depth_map = tensorf.forward_with_feature(rays_chunk, is_train=is_train, white_bg=white_bg,
                                                              ndc_ray=ndc_ray,
                                                              N_samples=N_samples, plane_feature=plane_feat_dict,
                                                              line_feature=batch_line_feature[b_idx],
                                                              alphaMask=batch_alpha_mask[b_idx])

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        del rgb_map
        del depth_map

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        # print('depth_map', depth_map.shape, depth_map.min(), depth_map.max())
        # print('rgb_map', rgb_map.shape, rgb_map.min(), rgb_map.max())
        # raise Exception('stop here')
        # depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far) # default
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), None) # changed @ 2024.4.10
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_gt.png', gt_rgb)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

