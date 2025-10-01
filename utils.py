import pdb

import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


from compressai.optimizers import net_aux_optimizer

def maybe_align_cdf_and_load_system(tensorf, system_ckpt_path):
    """
    Make sure all CompressAI decode-side buffers have the same shapes as the checkpoint
    before loading. This avoids size-mismatch errors across versions / init states.
    """
    system = torch.load(system_ckpt_path, map_location=device, weights_only=False)
    sd = system["state_dict"]

    # All decode-side buffers that can differ in shape / start empty:
    keys_to_fix = [
        # entropy bottleneck (hyperprior)
        "app_feat_codec.entropy_bottleneck._quantized_cdf",
        "app_feat_codec.entropy_bottleneck._cdf_length",
        "app_feat_codec.entropy_bottleneck._offset",
        "den_feat_codec.entropy_bottleneck._quantized_cdf",
        "den_feat_codec.entropy_bottleneck._cdf_length",
        "den_feat_codec.entropy_bottleneck._offset",
        # gaussian conditional (context+params)
        "app_feat_codec.gaussian_conditional._quantized_cdf",
        "app_feat_codec.gaussian_conditional._cdf_length",
        "app_feat_codec.gaussian_conditional._offset",
        "app_feat_codec.gaussian_conditional.scale_table",
        "den_feat_codec.gaussian_conditional._quantized_cdf",
        "den_feat_codec.gaussian_conditional._cdf_length",
        "den_feat_codec.gaussian_conditional._offset",
        "den_feat_codec.gaussian_conditional.scale_table",
    ]

    def _ensure_buffer_like(model, dotted_key, like_tensor):
        """
        Walk `model` along dotted_key (module path) and replace the *attribute* at the end
        with a new tensor of the right shape/dtype/device so load_state_dict won't complain.
        """
        parts = dotted_key.split(".")
        mod = model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        name = parts[-1]
        # keep dtype/device to match current module, but fall back to checkpoint dtype
        cur = getattr(mod, name, None)
        dtype = (cur.dtype if torch.is_tensor(cur) else like_tensor.dtype)
        dev   = (cur.device if torch.is_tensor(cur) else device)
        new_t = torch.zeros_like(like_tensor).to(dev, dtype=dtype)
        setattr(mod, name, new_t)

    # Pre-size any buffer whose size differs (or is missing) using the ckpt’s tensor
    for k in keys_to_fix:
        if k in sd:
            try:
                _ensure_buffer_like(tensorf, k, sd[k])
            except Exception:
                # safe to ignore if the module isn't present in this config
                pass

    # Now load everything (TensoRF + codec)
    tensorf.load(system)  # this calls load_state_dict(..., strict=False)

    # Important: refresh entropy tables to match current device / version
    if hasattr(tensorf, "app_feat_codec"):
        tensorf.app_feat_codec.update(force=True)
    if hasattr(tensorf, "den_feat_codec"):
        tensorf.den_feat_codec.update(force=True)

    return system

def load_optim_states_if_any(system, optimizer, aux_optimizer):
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

def make_model_ckpt_dict(tensorf):
    """Mirror tensorf.save(): pack model kwargs, state_dict, and alphaMask bundle."""
    kwargs = tensorf.get_kwargs()
    ckpt = {"kwargs": kwargs, "state_dict": tensorf.state_dict()}

    if getattr(tensorf, "alphaMask", None) is not None:
        alpha_volume = tensorf.alphaMask.alpha_volume.bool().cpu().numpy()
        ckpt["alphaMask.shape"] = alpha_volume.shape
        ckpt["alphaMask.mask"]  = np.packbits(alpha_volume.reshape(-1))
        ckpt["alphaMask.aabb"]  = tensorf.alphaMask.aabb.cpu()
    return ckpt

def save_system_ckpt(path, tensorf, optimizer, aux_optimizer, global_step, kwargs_override=None):
    """
    Save a resume-able checkpoint that still contains the same fields
    tensorf.save() used to write (incl. alphaMask) PLUS optimizers & step.
    """
    base = make_model_ckpt_dict(tensorf)

    # Optionally override kwargs (e.g., when saving right after building from args)
    if kwargs_override is not None:
        base["kwargs"] = kwargs_override

    base["optimizer"]     = optimizer.state_dict()     if optimizer     is not None else None
    base["aux_optimizer"] = aux_optimizer.state_dict() if aux_optimizer is not None else None
    base["global_step"]   = int(global_step)

    torch.save(base, path)

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.lr_feat_codec},
        "aux": {"type": "Adam", "lr": args.lr_aux},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    # import pdb
    # pdb.set_trace()
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    # print(f"rendered depth range:{ (mi, ma)}")

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

import math

def gaussian(x, mean, std_dev):
    """
    Args:
        x (torch.Tensor): input
        mean (float):
        std_dev (float):

    Returns:
        torch.Tensor:
    """
    coefficient = 1.0 / (std_dev * math.sqrt(2 * math.pi))
    exponent = -(x - mean) ** 2 / (2 * std_dev ** 2)
    prob_density = coefficient * torch.exp(exponent)

    return prob_density

def _standardized_cumulative(inputs: torch.Tensor) -> torch.Tensor:
    half = float(0.5)
    const = float(-(2**-0.5))
    # Using the complementary error function maximizes numerical precision.
    return half * torch.erfc(const * inputs)

from typing import Optional
from torch import Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


def likelihood(
    inputs: torch.Tensor,
    scales: torch.Tensor,
    means: Optional[torch.Tensor] = None
) -> torch.Tensor:

    half = 1 / float(2**8 - 1)

    # pdb.set_trace()
    if means is not None:
        values = inputs - means
    else:
        values = inputs
    lower_bound_scale = LowerBound(0.04).to(device)
    scales = lower_bound_scale(scales)

    values = torch.abs(values)
    upper = _standardized_cumulative((half - values) / scales)
    lower = _standardized_cumulative((-half - values) / scales)
    likelihood = upper - lower

    return likelihood

def features_rec_loss(tensorf, coding_output):
    map_fn = torch.nn.Tanh()
    # den
    den_feat_loss = 0
    for i in range(3):
        den_feat_loss += torch.mean(torch.abs(map_fn(tensorf.density_plane[i]) - coding_output["den"]["rec_planes"][i]))
    # app
    app_feat_loss = 0
    for i in range(3):
        app_feat_loss += torch.mean(torch.abs(map_fn(tensorf.app_plane[i]) - coding_output["app"]["rec_planes"][i]))

    feat_rec_loss = den_feat_loss + app_feat_loss

    return feat_rec_loss