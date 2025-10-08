# --- keep other imports in the file as-is ---
import torch
import torch.nn.functional as F
import math
from compressai.ops import compute_padding

from .tensorBase import TensorBase
from .qat import qfn2  # only used if vec_qat is enabled


class TensorVMSplit(TensorBase):
    """
    Split TensoRF (vector+matrix factorization) with optional adaptor feature-codec.

    This pruned version keeps only what the simplified training script uses:
      - triplane parameters (density_plane/line, app_plane/line) and rendering
      - adaptor feature codec (den_feat_codec/app_feat_codec) initialization, warmup, rate (aux) loss
      - (optional) 'additional_vec' support (off by default)
      - upsample/shrink + regularizers used by the trainer

    Removed:
      - image-codec path and 'batchwise_img_coding' strategy
      - generic forward_with_feature() path
      - miscellaneous ckpt helpers for image codec
    """

    # ----------------------------------------------------------------------------------
    # Lifecyle / flags
    # ----------------------------------------------------------------------------------
    def __init__(self, aabb, gridSize, device, **kargs):
        """
        Construct triplane volumes and rendering MLP (done by TensorBase).
        Adds flags that the trainer uses to enable the adaptor feature codec flow.
        """
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

        # codec-related flags (used by simplified train.py)
        self.compression = False
        self.compression_strategy = kargs.get('compression_strategy', None)  # we only use 'adaptor_feat_coding'
        self.compress_before_volrend = kargs.get('compress_before_volrend', False)
        self.mode = "train"

        # generic switches supported by trainer (usually False in your runs)
        self.using_external_codec = False          # treat pre-compressed planes as inputs
        self.additional_vec = False                # optional extra line factors
        self.vec_qat = kargs.get('vec_qat', False)
        self.decode_from_latent_code = kargs.get('decode_from_latent_code', False)

    # ----------------------------------------------------------------------------------
    # Adaptor feature codec (the path you use)
    # ----------------------------------------------------------------------------------
    def init_feat_codec(
        self,
        codec_ckpt_path: str = '',
        loading_pretrain_param: bool = True,
        adaptor_q_bit: int = 8,
        codec_backbone_type: str = "cheng2020-anchor",
    ):
        """
        Initialize *two* adaptors (neural codec backbones from CompressAI, wrapped by our Adaptor):
          - den_feat_codec: for density planes (channels = density_n_comp[0])
          - app_feat_codec: for appearance planes (channels = app_n_comp[0])

        Called once in _build_model(...) when --compression and adaptor_feat_coding are enabled.
        """
        from .imageCoder import (
            AdaptorScaleHyperprior, AdaptorMeanScaleHyperprior,
            AdaptorCheng2020Anchor, AdaptorCheng2020Attention,
        )
        from compressai.zoo.image import model_urls, cfgs
        from compressai.zoo.pretrained import load_pretrained
        from torch.hub import load_state_dict_from_url

        feat_codec_dict = {
            "bmshj2018-hyperprior": AdaptorScaleHyperprior,
            "mbt2018-mean":         AdaptorMeanScaleHyperprior,
            "cheng2020-anchor":     AdaptorCheng2020Anchor,
            "cheng2020-attn":       AdaptorCheng2020Attention,
        }

        architecture = codec_backbone_type
        metric = "mse"
        # your runs use cheng2020 with quality=6
        quality = 6

        if self.decode_from_latent_code:
            raise NotImplementedError("decode_from_latent_code path retained in simplified train.py, "
                                      "but decoder-only adaptor is not implemented here.")

        feat_codec = feat_codec_dict[architecture]

        # load a pretrained image model's weights as initializer for the adaptors
        if codec_ckpt_path == "":
            url = model_urls[architecture][metric][quality]
            state = load_state_dict_from_url(url, progress=True)
            state = load_pretrained(state)
        else:
            codec_ckpt = torch.load(codec_ckpt_path)

        self.latent_code_ch = cfgs[architecture][quality][-1]

        # density adaptor (channels per plane = self.density_n_comp[0])
        self.den_feat_codec = feat_codec(self.density_n_comp[0], *cfgs[architecture][quality], q_bit=adaptor_q_bit)
        if loading_pretrain_param:
            if codec_ckpt_path == "":
                self.den_feat_codec.load_state_dict(state, strict=False)
                self.den_feat_codec.reload_from_pretrained()  # also loads priors
            else:
                self.den_feat_codec.load_state_dict(codec_ckpt["den_feat_codec"])
        self.den_feat_codec.to(self.device)

        # appearance adaptor (channels per plane = self.app_n_comp[0])
        self.app_feat_codec = feat_codec(self.app_n_comp[0], *cfgs[architecture][quality], q_bit=adaptor_q_bit)
        if loading_pretrain_param:
            if codec_ckpt_path == "":
                self.app_feat_codec.load_state_dict(state, strict=False)
                self.app_feat_codec.reload_from_pretrained()
            else:
                self.app_feat_codec.load_state_dict(codec_ckpt["app_feat_codec"])
        self.app_feat_codec.to(self.device)

        self.compression = True

        # latent-code path exists in trainer as a toggle, keep helpers initialized if needed
        if self.decode_from_latent_code:
            self.latent_z_size = torch.ceil(self.gridSize / 64.0)
            self.latent_y_size = self.latent_z_size * 4
            self.den_latent_y, self.den_latent_z = self.init_one_latent_code(
                self.latent_y_size, self.latent_z_size, 0.1, self.device
            )
            self.app_latent_y, self.app_latent_z = self.init_one_latent_code(
                self.latent_y_size, self.latent_z_size, 0.1, self.device
            )

    def get_optparam_from_feat_codec(self, lr_transform=1e-4, fix_decoder_prior=False, fix_encoder_prior=False):
        """
        At warmup phase, lr_enc_prior = 0, so all “prior” params are frozen. Only the adaptor params train
        """
        params_dict = dict(self.named_parameters())

        adaptor_param_list = {
            name for name, p in self.named_parameters()
            if p.requires_grad and 'adaptor' in name
        }
        prior_param_list = {
            name for name, p in self.named_parameters()
            if p.requires_grad and ('codec' in name) and (not name.endswith(".quantiles")) and ('adaptor' not in name)
        }
        aux_param_list = {
            name for name, p in self.named_parameters()
            if p.requires_grad and name.endswith(".quantiles")
        }

        if fix_decoder_prior:
            prior_param_list = [n for n in prior_param_list if ('g_s' not in n and 'h_s' not in n)]

        lr_enc_prior = 0 if fix_encoder_prior else 1e-5
        print("priors in enc. is fixed" if lr_enc_prior == 0 else "priors in enc. is not fixed")

        grad_vars = [
            {'params': (params_dict[n] for n in sorted(adaptor_param_list)), 'lr': lr_transform},
            {'params': (params_dict[n] for n in sorted(prior_param_list)),   'lr': lr_enc_prior},
        ]
        aux_grad_vars = [
            {'params': (params_dict[n] for n in sorted(aux_param_list)), 'lr': 1e-3},
        ]
        return grad_vars, aux_grad_vars

    def save_feat_codec_ckpt(self, ckpt_dir):
        """Utility; not used by simplified train loop but harmless to keep."""
        torch.save(self.app_feat_codec.state_dict(), f"{ckpt_dir}/app_codec.ckpt")
        torch.save(self.den_feat_codec.state_dict(), f"{ckpt_dir}/den_codec.ckpt")

    def load_feat_codec_ckpt(self, ckpt_dir):
        """Utility; not used by simplified train loop but harmless to keep."""
        self.app_feat_codec.load_state_dict(torch.load(f"{ckpt_dir}/app_codec.ckpt"))
        self.den_feat_codec.load_state_dict(torch.load(f"{ckpt_dir}/den_codec.ckpt"))

    def get_aux_loss(self):
        """
        Aux loss for entropy models (quantile loss).
        Called every iteration in the codec finetune stage (aux optimizer step).
        """
        assert self.compression_strategy == "adaptor_feat_coding", \
            "Pruned TensorVMSplit only supports adaptor_feat_coding"
        return self.den_feat_codec.aux_loss() + self.app_feat_codec.aux_loss()

    # ----------------------------------------------------------------------------------
    # Triplane parameters (init/upsample/shrink) + optional extras
    # ----------------------------------------------------------------------------------
    def init_svd_volume(self, res, device):
        """
        Allocate triplane parameters:
          - density_plane/line and app_plane/line (three planes + three lines)
          - basis_mat to fuse app features into app_dim (for shading)
        Called by TensorBase.__init__ at construction.
        """
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane,     self.app_line     = self.init_one_svd(self.app_n_comp,     self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """Helper: create 3 plane tensors and 3 line tensors for a triplane factorization."""
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])))
            )
            line_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[vec_id], 1)))
            )
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def init_one_latent_code(self, y_gridSize, z_gridSize, scale, device):
        """
        Optional latent-code path (kept since simplified train.py still toggles it).
        Not used in your standard runs.
        """
        latent_code_y, latent_code_z = [], []
        for i in range(len(self.vecMode)):
            m0, m1 = self.matMode[i]
            latent_code_y.append(torch.nn.Parameter(
                scale * torch.randn((1, self.latent_code_ch, int(y_gridSize[m1]), int(y_gridSize[m0])))))
            latent_code_z.append(torch.nn.Parameter(
                scale * torch.randn((1, self.latent_code_ch, int(z_gridSize[m1]), int(z_gridSize[m0])))))
        return torch.nn.ParameterList(latent_code_y).to(device), torch.nn.ParameterList(latent_code_z).to(device)

    def init_additional_volume(self, device):
        """
        Optional extra 'line' features along each axis.
        Only used if args.additional_vec is True.
        """
        self.additional_vec = True
        self.additional_density_line = self.init_additional_svd(self.density_n_comp, self.gridSize * 2, 0.1, device)
        self.additional_app_line     = self.init_additional_svd(self.app_n_comp,     self.gridSize * 2, 0.1, device)

    def init_additional_svd(self, n_component, gridSize, scale, device):
        """Helper for the optional additional_vec path."""
        line_coef = []
        for i in range(len(self.vecMode)):
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[i], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001, fix_plane=False):
        """
        Collect trainable param groups for the main optimizer.
        Used both in pretraining and codec finetuning.

        If fix_plane=True, planes are frozen (only lines+basis+MLP get updated).
        """
        grad_vars = [{'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        lr_plane = 0 if fix_plane else lr_init_spatialxyz
        grad_vars += [
            {'params': self.density_line, 'lr': lr_init_spatialxyz},
            {'params': self.density_plane, 'lr': lr_plane},
            {'params': self.app_line,     'lr': lr_init_spatialxyz},
            {'params': self.app_plane,    'lr': lr_plane},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def get_additional_optparam_groups(self, lr_init_spatialxyz=0.02):
        """Optimizer params for the optional additional_vec path."""
        return [
            {'params': self.additional_density_line, 'lr': lr_init_spatialxyz},
            {'params': self.additional_app_line,     'lr': lr_init_spatialxyz},
        ]

    def get_latent_code_groups(self, lr_latent_code=0.002):
        """Optimizer params for the optional latent-code path (not used in your standard runs)."""
        return [
            {'params': self.den_latent_y, 'lr': lr_latent_code},
            {'params': self.den_latent_z, 'lr': lr_latent_code},
            {'params': self.app_latent_y, 'lr': lr_latent_code},
            {'params': self.app_latent_z, 'lr': lr_latent_code},
        ]

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        """
        Bilinear upsample planes/lines to a new triplane resolution.
        Called inside upsample_volume_grid(...) during progressive training.
        """
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            m0, m1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[m1], res_target[m0]),
                              mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1),
                              mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        """
        Public API used by the trainer’s upsample schedule.
        """
        self.app_plane,     self.app_line     = self.up_sampling_VM(self.app_plane,     self.app_line,     res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        """
        Crop the triplane tensors to a new AABB computed from the alpha mask.
        Called by the trainer right after first alpha-mask creation.
        """
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            self.app_line[i]     = torch.nn.Parameter(self.app_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])

        # correct aabb if alpha volume grid differs from current gridSize
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    # ----------------------------------------------------------------------------------
    # Rendering-time feature gathering (used by TensorBase.forward)
    # ----------------------------------------------------------------------------------
    def enable_vec_qat(self):
        """
        Enable/disable vector quantization-aware training (line features only).
        Trainer calls this once after building the model.
        """
        if self.vec_qat:
            print("Vector, QAT, bitwidth: 8 bits")
            self.q_fn = lambda x: qfn2.apply(x, 8)
        else:
            self.q_fn = torch.nn.Identity()

    def compress_with_external_codec(self, den_feat_codec, app_feat_codec, mode="train"):
        """
        Compress (or pseudo-compress during training) each plane with the given codec(s),
        and store the reconstructed planes in self.{den,app}_rec_plane (list of 3 tensors).
        The trainer calls this at each step when --compress_before_volrend is set.

        Returns a dict with reconstructed planes + likelihood packs (for rate / aux).
        """
        self.map_fn = torch.nn.Tanh()

        # density planes
        self.den_rec_plane, self.den_likelihood = [], []
        for i in range(len(self.density_plane)):
            rec_plane, likelihood = self.feature_compression_via_feat_coder(
                self.map_fn(self.density_plane[i]), den_feat_codec, mode=mode)
            self.den_rec_plane.append(rec_plane)
            self.den_likelihood.append(likelihood)

        # appearance planes
        self.app_rec_plane, self.app_likelihood = [], []
        for i in range(len(self.app_plane)):
            rec_plane, likelihood = self.feature_compression_via_feat_coder(
                self.map_fn(self.app_plane[i]), app_feat_codec, mode=mode)
            self.app_rec_plane.append(rec_plane)
            self.app_likelihood.append(likelihood)

        return {
            "den": {"rec_planes": self.den_rec_plane, "rec_likelihood": self.den_likelihood},
            "app": {"rec_planes": self.app_rec_plane, "rec_likelihood": self.app_likelihood},
        }


    def feature_compression_via_feat_coder(self, plane, feat_codec, return_likelihoods=True, mode="train"):
        """
        Core 'adaptor_feat_coding' op:
          - per-plane min-max normalize
          - pad spatially to 64-aligned size
          - run codec.forward() during training, codec.compress/decompress() during eval
          - unpad and denormalize

        Used by compress_with_external_codec() and by latent-code decoder path.
        """
        # per-plane min/max over spatial dims
        min_vec, _ = torch.min(plane.view(*plane.shape[:2], -1), dim=-1)
        max_vec, _ = torch.max(plane.view(*plane.shape[:2], -1), dim=-1)
        min_vec, max_vec = min_vec.view([*min_vec.shape, 1, 1]), max_vec.view([*max_vec.shape, 1, 1])
        norm_plane = (plane - min_vec) / (max_vec - min_vec + 1e-8)

        # pad to multiples of 2^6
        h, w = norm_plane.size(2), norm_plane.size(3)
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)
        norm_plane_padded = F.pad(norm_plane, pad, mode="constant", value=0)

        if mode == "train":
            out_net = feat_codec.forward(norm_plane_padded)
        else:
            out_enc = feat_codec.compress(norm_plane_padded)

            # --- robust byte counter for nested lists/tuples of byte strings ---
            def _count_bytes(obj):
                try:
                    import torch, numpy as np
                except Exception:
                    torch, np = None, None

                # bytes-like
                if isinstance(obj, (bytes, bytearray, memoryview)):
                    return len(obj)

                # nested containers
                if isinstance(obj, (list, tuple)):
                    return sum(_count_bytes(x) for x in obj)

                # python str (some compressai builds return str for y_strings)
                if isinstance(obj, str):
                    # latin-1 preserves length 1:1 for 0..255
                    try:
                        return len(obj.encode("latin-1", errors="ignore"))
                    except Exception:
                        return len(obj)

                # torch uint8 tensor
                if torch is not None and torch.is_tensor(obj) and obj.dtype == torch.uint8:
                    return int(obj.numel())

                # numpy uint8 array
                if (("np" in locals() and np is not None)
                        and isinstance(obj, np.ndarray)
                        and obj.dtype == np.uint8):
                    return int(obj.size)

                raise ValueError(f"Cannot count bytes of object type {type(obj)}")
                return 0
            
            total_bytes = _count_bytes(out_enc["strings"])

            out_dec = feat_codec.decompress(out_enc["strings"], out_enc["shape"])
            out_net = out_dec
            out_net["strings_length_bytes"] = int(total_bytes)

        # unpad + denorm
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_net["x_hat"] = out_net["x_hat"].reshape([1, -1, h, w])
        rec_plane = out_net["x_hat"] * (max_vec - min_vec + 1e-8) + min_vec

        rec_plane = torch.nan_to_num(rec_plane, nan=0.0, posinf=0.0, neginf=0.0)

        return (rec_plane, out_net) if return_likelihoods else (rec_plane, None)

    def decode_single_plane(self, y, z, feat_codec, target_plane_size, mode="train"):
        """
        Latent-code variant: decode one plane from (y,z) latents using the adaptor.
        Kept because simplified train.py still toggles --decode_from_latent_code.
        """
        h, w = target_plane_size[2], target_plane_size[3]
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)

        if mode == "train":
            out_net = feat_codec.forward(y, z)
        else:
            out_enc = feat_codec.compress(y, z)
            out_dec = feat_codec.decompress(out_enc["strings"], out_enc["shape"])
            out_net = out_dec
            out_net.update({"strings_length": sum(len(s[0]) for s in out_enc["strings"])})

        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_net["x_hat"] = out_net["x_hat"].reshape([1, -1, h, w])
        return out_net

    def decode_all_planes(self, mode="train"):
        """
        Latent-code variant: decode all 3 planes for density and appearance.
        Returns same structure as compress_with_external_codec(...).
        """
        self.map_fn = torch.nn.Tanh()
        self.den_rec_plane, self.den_likelihood = [], []
        for i in range(len(self.density_plane)):
            out_net = self.decode_single_plane(
                self.den_latent_y[i], self.den_latent_z[i], self.den_feat_codec,
                self.density_plane[i].size(), mode=mode
            )
            self.den_rec_plane.append(out_net["x_hat"])
            self.den_likelihood.append(out_net)

        self.app_rec_plane, self.app_likelihood = [], []
        for i in range(len(self.app_plane)):
            out_net = self.decode_single_plane(
                self.app_latent_y[i], self.app_latent_z[i], self.app_feat_codec,
                self.app_plane[i].size(), mode=mode
            )
            self.app_rec_plane.append(out_net["x_hat"])
            self.app_likelihood.append(out_net)

        return {
            "den": {"rec_planes": self.den_rec_plane, "rec_likelihood": self.den_likelihood},
            "app": {"rec_planes": self.app_rec_plane, "rec_likelihood": self.app_likelihood},
        }

    # ----------------------------------------------------------------------------------
    # Feature sampling for volume rendering
    # ----------------------------------------------------------------------------------
    def compute_densityfeature(self, xyz_sampled):
        """
        Sample density feature at normalized coords and fuse plane/line factors.

        Two modes:
          - compression+compress_before_volrend: read plane coeffs from reconstructed planes
          - else: read plane coeffs from raw learnable planes

        Called inside TensorBase.forward(...) during both training and evaluation.
        """
        self.map_fn = torch.nn.Identity()  # keep density planes unbounded (match original)

        # build sampling coords for the three planes and three lines
        coordinate_plane = torch.stack((
            xyz_sampled[..., self.matMode[0]],
            xyz_sampled[..., self.matMode[1]],
            xyz_sampled[..., self.matMode[2]],
        )).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((
            xyz_sampled[..., self.vecMode[0]],
            xyz_sampled[..., self.vecMode[1]],
            xyz_sampled[..., self.vecMode[2]],
        ))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        if self.additional_vec:
            add_coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
            add_coordinate_line = torch.stack((torch.zeros_like(add_coordinate_line), add_coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for i in range(len(self.density_plane)):
            # choose plane source: reconstructed or raw learnable
            if self.compression and (self.compress_before_volrend or self.using_external_codec):
                if self._noise_enabled():
                    plane_src = self._noise_op(
                        self.density_plane[i], 
                        self.noise_cfg["mode"], 
                        self.noise_cfg["level"], 
                        self.noise_cfg["seed"], 
                        plane_idx=i
                    )
                else:
                    plane_src = self.den_rec_plane[i]
            else:
                plane_src = self.map_fn(self.density_plane[i])

            plane_coef_point = F.grid_sample(plane_src, coordinate_plane[[i]], align_corners=True).view(-1, *xyz_sampled.shape[:1])

            if self.additional_vec:
                m0, m1 = self.matMode[i]
                fst_add = F.grid_sample(self.q_fn(self.map_fn(self.additional_density_line[m0])),
                                        add_coordinate_line[[m0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
                sec_add = F.grid_sample(self.q_fn(self.map_fn(self.additional_density_line[m1])),
                                        add_coordinate_line[[m1]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
                plane_coef_point += (fst_add * sec_add)

            # line factors (optionally QAT-ed)
            if self.compression and self.vec_qat:
                line_src = self.q_fn(self.map_fn(self.density_line[i]))
            else:
                line_src = self.map_fn(self.density_line[i])
            line_coef_point = F.grid_sample(line_src, coordinate_line[[i]], align_corners=True).view(-1, *xyz_sampled.shape[:1])

            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        """
        Sample appearance feature at normalized coords and fuse via basis_mat.

        Same source switching as density: reconstructed planes if we compressed before volume
        rendering; otherwise use raw learnable planes. App planes pass through tanh as in original.
        """
        self.map_fn = torch.nn.Tanh()

        coordinate_plane = torch.stack((
            xyz_sampled[..., self.matMode[0]],
            xyz_sampled[..., self.matMode[1]],
            xyz_sampled[..., self.matMode[2]],
        )).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((
            xyz_sampled[..., self.vecMode[0]],
            xyz_sampled[..., self.vecMode[1]],
            xyz_sampled[..., self.vecMode[2]],
        ))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        if self.additional_vec:
            add_coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
            add_coordinate_line = torch.stack((torch.zeros_like(add_coordinate_line), add_coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for i in range(len(self.app_plane)):
            if self.compression and (self.compress_before_volrend or self.using_external_codec):
                if self._noise_enabled():
                    plane_src = self._noise_op(
                        torch.tanh(self.app_plane[i]),
                        self.noise_cfg["mode"],
                        self.noise_cfg["level"],
                        self.noise_cfg["seed"],
                        plane_idx=100 + i  # different stream
                    )
                else:
                    plane_src = self.app_rec_plane[i]
            else:
                plane_src = self.map_fn(self.app_plane[i])

            p = F.grid_sample(plane_src, coordinate_plane[[i]], align_corners=True).view(-1, *xyz_sampled.shape[:1])

            if self.additional_vec:
                m0, m1 = self.matMode[i]
                fst_add = F.grid_sample(self.q_fn(self.map_fn(self.additional_app_line[m0])),
                                        add_coordinate_line[[m0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
                sec_add = F.grid_sample(self.q_fn(self.map_fn(self.additional_app_line[m1])),
                                        add_coordinate_line[[m1]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
                p += (fst_add * sec_add)

            plane_coef_point.append(p)

            if self.compression and self.vec_qat:
                line_src = self.q_fn(self.map_fn(self.app_line[i]))
            else:
                line_src = self.map_fn(self.app_line[i])
            line_coef_point.append(F.grid_sample(line_src, coordinate_line[[i]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point  = torch.cat(line_coef_point)
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    # ----------------------------------------------------------------------------------
    # Regularizers (used by trainer)
    # ----------------------------------------------------------------------------------
    def vectorDiffs(self, vector_comps):
        """Mutual orthogonality penalty among line vectors (used by trainer with Ortho_weight)."""
        total = 0
        for i in range(len(vector_comps)):
            n_comp, n_size = vector_comps[i].shape[1:-1]
            dotp = torch.matmul(vector_comps[i].view(n_comp, n_size),
                                vector_comps[i].view(n_comp, n_size).transpose(-1, -2))
            non_diag = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total += torch.mean(torch.abs(non_diag))
        return total

    def vector_comp_diffs(self):
        """Sum orthogonality penalty over density and appearance line sets."""
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        """L1 penalty over density planes and lines (used during early training)."""
        total = 0
        for i in range(len(self.density_plane)):
            total += torch.mean(torch.abs(self.density_plane[i])) + torch.mean(torch.abs(self.density_line[i]))
        return total

    def TV_loss_density(self, reg):
        """Total variation on density planes (called with scaling in trainer)."""
        total = 0
        for i in range(len(self.density_plane)):
            total += reg(self.density_plane[i]) * 1e-2
        return total

    def TV_loss_app(self, reg):
        """Total variation on appearance planes (called with scaling in trainer)."""
        total = 0
        for i in range(len(self.app_plane)):
            total += reg(self.app_plane[i]) * 1e-2
        return total
    
    def estimate_codec_transmission_bits(
        self,
        mode: str = "raw",            # "raw" or "quant-ent"
        q_bits: int = 8,              # used only for "quant-ent"
        include_header: bool = True,  # per-tensor header when "quant-ent"
        return_breakdown: bool = True # include per-module detail
    ):
        """
        What we transmit to the client (decoder side only):

        Codec (both density & appearance):
            - decoder_adaptor
            - context_prediction
            - entropy_parameters
            - entropy_bottleneck.quantiles     (learned; receiver calls .update() to rebuild CDFs)

        Neural renderer:
            - basis_mat                        (feature fusion)
            - renderModule                     (MLP head, e.g., MLPRender_Fea/PE)

        Frozen priors (e.g., trunc_g_s / h_s when fix_decoder_prior=True) and all encoder-side parts
        are excluded by design.

        Units: returns both bits and MB (MB = bytes / 1e6).
        """

        import math
        import torch

        assert mode in ("raw", "quant-ent"), "mode must be 'raw' or 'quant-ent'"
        assert hasattr(self, "den_feat_codec") and hasattr(self, "app_feat_codec"), \
            "Feature codecs are not initialized. Call init_feat_codec(...) first."

        # ---------- per-tensor size models ----------
        def _tensor_bits_raw(t: torch.Tensor) -> int:
            return int(t.numel() * t.element_size() * 8)

        def _tensor_bits_quant(t: torch.Tensor) -> float:
            # Uniform quantize to q_bits, Shannon bound over empirical PMF + tiny header
            x = t.detach().float().cpu()
            N = x.numel()
            if N == 0:
                H_bits = 0.0
            else:
                xmin = float(torch.min(x))
                xmax = float(torch.max(x))
                if (not math.isfinite(xmin)) or (not math.isfinite(xmax)) or (xmax - xmin) <= 0:
                    H_bits = 0.0
                else:
                    L = 2 ** q_bits
                    scale = (xmax - xmin) / (L - 1)
                    q = torch.round((x - xmin) / (scale if scale > 0 else 1.0)).clamp_(0, L - 1).to(torch.int64)
                    hist = torch.bincount(q.view(-1), minlength=L).float()
                    probs = hist / float(N)
                    nz = probs > 0
                    H = -(probs[nz] * torch.log2(probs[nz])).sum().item()
                    H_bits = H * float(N)
            header_bits = 0
            if include_header:
                header_bits += 32 * 2              # min,max as float32
                header_bits += 8                   # q_bits as uint8
                header_bits += 32 * len(t.shape)   # int32 per dim
            return float(H_bits + header_bits)

        def _list_bits_raw(tensors):   return sum(_tensor_bits_raw(p) for p in tensors)
        def _list_bits_quant(tensors): return float(sum(_tensor_bits_quant(p) for p in tensors))

        def _sum_group(tensors_by_name):
            if mode == "raw":
                bits = {k: _list_bits_raw(v)   for k, v in tensors_by_name.items()}
            else:
                bits = {k: _list_bits_quant(v) for k, v in tensors_by_name.items()}
            total_bits = float(sum(bits.values()))
            return bits, total_bits

        # ---------- collect exactly the decode-side, learned parts ----------
        def _collect_codec_parts(codec):
            parts = {
                "decoder_adaptor":      list(codec.decoder_adaptor.parameters())     if hasattr(codec, "decoder_adaptor") else [],
                "context_prediction":   list(codec.context_prediction.parameters())  if hasattr(codec, "context_prediction") else [],
                "entropy_parameters":   list(codec.entropy_parameters.parameters())  if hasattr(codec, "entropy_parameters") else [],
                "entropy_bottleneck.quantiles": []
            }
            for n, p in codec.named_parameters():
                if n.endswith("quantiles"):
                    parts["entropy_bottleneck.quantiles"].append(p)
            return parts

        den_parts = _collect_codec_parts(self.den_feat_codec)
        app_parts = _collect_codec_parts(self.app_feat_codec)

        den_bits_by_mod, den_total_bits = _sum_group(den_parts)
        app_bits_by_mod, app_total_bits = _sum_group(app_parts)

        # ---------- renderer payload (basis_mat + renderModule) ----------
        renderer_parts = {
            "basis_mat": list(self.basis_mat.parameters()) if hasattr(self, "basis_mat") else [],
            "render_mlp": list(self.renderModule.parameters()) if isinstance(self.renderModule, torch.nn.Module) else [],
        }
        rend_bits_by_mod, rend_total_bits = _sum_group(renderer_parts)

        grand_total_bits = den_total_bits + app_total_bits + rend_total_bits

        # ---------- also report MB (decimal) ----------
        def _bits_to_MB(bits: float) -> float:
            return float(bits) / 8.0 / 1e6

        out = {
            "mode": mode,
            "q_bits": q_bits if mode == "quant-ent" else None,
            "include_header": include_header if mode == "quant-ent" else None,

            # Codec totals
            "density_codec": {
                "total_bits": den_total_bits,
                "total_MB": _bits_to_MB(den_total_bits),
                **({"breakdown_bits": den_bits_by_mod,
                    "breakdown_MB": {k: _bits_to_MB(v) for k, v in den_bits_by_mod.items()}} if return_breakdown else {})
            },
            "appearance_codec": {
                "total_bits": app_total_bits,
                "total_MB": _bits_to_MB(app_total_bits),
                **({"breakdown_bits": app_bits_by_mod,
                    "breakdown_MB": {k: _bits_to_MB(v) for k, v in app_bits_by_mod.items()}} if return_breakdown else {})
            },

            # Renderer totals
            "renderer": {
                "total_bits": rend_total_bits,
                "total_MB": _bits_to_MB(rend_total_bits),
                **({"breakdown_bits": rend_bits_by_mod,
                    "breakdown_MB": {k: _bits_to_MB(v) for k, v in rend_bits_by_mod.items()}} if return_breakdown else {})
            },

            # Grand total
            "total_bits_all": grand_total_bits,
            "total_MB_all": _bits_to_MB(grand_total_bits),
        }
        return out
    
    # In class TensorVMSplit -----------------------------------------------

    def _ste(self, x, fwd):
        """
        Straight-through wrapper: forward uses fwd(x) (non-diff),
        backward is identity wrt x.
        """
        with torch.no_grad():
            y = fwd(x)
        return x + (y - x).detach()
    
    def _noise_op(self, plane, mode, level, seed, plane_idx):
        """
        Deterministic noise operator under STE.

        Modes:
        - "quant"    : uniform quant/dequant within FIXED global range
        - "gaussian" : additive N(0, sigma% * FIXED range), clamp to FIXED range
        - "blocky"   : depthwise box blur (k×k), then uniform quant within FIXED range

        Fixed global ranges (STE+JPEG style):
        density planes (plane_idx < 100):  [-25, 25]
        appearance planes (plane_idx >=100): [-5, 5]

        Levels (int): 1=light, 2=medium, 3=heavy (clamped to [1,3])
        """
        dev   = plane.device
        dtype = plane.dtype

        # clamp level
        try:
            lvl = int(level)
        except Exception:
            lvl = 1
        lvl = 1 if lvl < 1 else (3 if lvl > 3 else lvl)

        # per-device generator, deterministic per (seed, plane_idx) for gaussian
        g = torch.Generator(device=dev.type)
        g.manual_seed(int(seed) ^ (int(plane_idx) * 0x9e3779b1))

        # -------- fixed global ranges ----------
        is_app   = (int(plane_idx) >= 100)
        rng_lo   = -5.0 if is_app else -25.0
        rng_hi   =  5.0 if is_app else  25.0
        rng_span = (rng_hi - rng_lo)  # 10 or 50

        # ---------------- QUANT ----------------
        if mode == "quant":
            # 1/2/3 → 8 / 4 / 2 bits
            bits = {1: 8, 2: 4, 3: 2}[lvl]
            L    = 2 ** bits

            def _q(x):
                with torch.no_grad():
                    # normalize to [0,1] by FIXED global range
                    xn = (x - rng_lo) / (rng_span + 1e-8)
                    xn = xn.clamp(0.0, 1.0)
                    q  = torch.round(xn * (L - 1)) / (L - 1)
                    y  = q * rng_span + rng_lo
                return y

            return self._ste(plane, _q)

        # --------------- GAUSSIAN ---------------
        elif mode == "gaussian":
            # 1/2/3 → sigma_pct = 2% / 5% / 10% of the FIXED range
            sigma = {1: 0.05, 2: 0.10, 3: 0.20}[lvl] * rng_span

            def _add(x):
                with torch.no_grad():
                    noise = torch.normal(
                        mean=0.0, std=1.0, size=x.shape, generator=g, device=dev, dtype=dtype
                    ) * sigma
                    y = (x + noise).clamp(rng_lo, rng_hi)
                return y

            return self._ste(plane, _add)

        # ---------------- BLOCKY ----------------
        elif mode == "blocky":
            # lowpass_q: blur + uniform quant (all in FIXED range)
            # 1: k=3,  q=6 bits   | 2: k=5, q=4 bits   | 3: k=9, q=3 bits
            k, q_bits = {1: (3, 8), 2: (5, 8), 3: (7, 6)}[lvl]
            k = max(1, min(k, plane.shape[-2], plane.shape[-1]))
            L = 2 ** q_bits

            kernel = torch.ones(1, 1, k, k, device=dev, dtype=dtype) / float(k * k)

            def _lpq(x):
                with torch.no_grad():
                    # depthwise blur
                    C = x.shape[1]
                    x_blur = torch.nn.functional.conv2d(
                        x, kernel.expand(C, 1, k, k), padding=k // 2, groups=C
                    )
                    # quantize in FIXED global range
                    xn = (x_blur - rng_lo) / (rng_span + 1e-8)
                    xn = xn.clamp(0.0, 1.0)
                    q  = torch.round(xn * (L - 1)) / (L - 1)
                    y  = q * rng_span + rng_lo
                return y

            return self._ste(plane, _lpq)

                # ---------------- CODEC_BLOCK ----------------
        elif mode == "codec_block":
            # block-average -> fixed-range quant on block DC -> NN upsample
            # 1/2/3 → (k, q_bits) = (4,8) / (8,6) / (16,4)
            k, q_bits = {1: (2, 8), 2: (4, 6), 3: (8, 4)}[lvl]
            # keep k valid for small planes
            k = max(1, min(k, plane.shape[-2], plane.shape[-1]))
            if k <= 1:
                return plane

            L = 2 ** q_bits

            def _codec_block(x):
                with torch.no_grad():
                    # (1) non-overlapping block-average
                    pooled = torch.nn.functional.avg_pool2d(
                        x, kernel_size=k, stride=k, ceil_mode=False
                    )
                    # (2) quantize pooled values in FIXED global range
                    xn = (pooled - rng_lo) / (rng_span + 1e-8)
                    xn = xn.clamp(0.0, 1.0)
                    q  = torch.round(xn * (L - 1)) / (L - 1)
                    pooled_q = q * rng_span + rng_lo
                    # (3) upsample back with NN (keeps block edges sharp)
                    up = torch.nn.functional.interpolate(
                        pooled_q, size=x.shape[-2:], mode="nearest"
                    )
                    return up

            return self._ste(plane, _codec_block)


        # fallback
        return plane


    def _noise_enabled(self):
        return bool(getattr(self, "noise_cfg", {"enabled": False}).get("enabled", False))

    # def compress_with_noise(self, mode="train"):
    #     """
    #     Mirror compress_with_external_codec but apply deterministic noise instead of a codec.
    #     Writes self.den_rec_plane / self.app_rec_plane and returns a matching dict.
    #     """
    #     assert getattr(self, "noise_cfg", {"enabled": False})["enabled"], \
    #         "compress_with_noise called but noise_cfg is disabled."

    #     self._release_noise_cache()

    #     self.map_fn = torch.nn.Tanh()  # match your app path; density uses Identity later

    #     cfg = self.noise_cfg
    #     mode_ = cfg["mode"]; lvl = cfg["level"]; seed = cfg["seed"]

    #     # density planes (use Identity like your compute_densityfeature)
    #     self.den_rec_plane, self.den_likelihood = [], []
    #     for i in range(len(self.density_plane)):
    #         src = self.density_plane[i]  # leave unbounded; match compute_densityfeature
    #         rec = self._noise_op(src, mode_, lvl, seed, plane_idx=i)
    #         self.den_rec_plane.append(rec)
    #         # Minimal, rate-less placeholder to keep trainer happy
    #         self.den_likelihood.append({"likelihoods": {"y": torch.ones(1, device=src.device),
    #                                                     "z": torch.ones(1, device=src.device)},
    #                                     "strings_length_bytes": 0})

    #     # appearance planes (tanh like your normal path)
    #     self.app_rec_plane, self.app_likelihood = [], []
    #     for i in range(len(self.app_plane)):
    #         src = torch.tanh(self.app_plane[i])
    #         rec = self._noise_op(src, mode_, lvl, seed, plane_idx=100 + i)  # different seed stream
    #         self.app_rec_plane.append(rec)
    #         self.app_likelihood.append({"likelihoods": {"y": torch.ones(1, device=src.device),
    #                                                     "z": torch.ones(1, device=src.device)},
    #                                     "strings_length_bytes": 0})

    #     return {
    #         "den": {"rec_planes": self.den_rec_plane, "rec_likelihood": self.den_likelihood},
    #         "app": {"rec_planes": self.app_rec_plane, "rec_likelihood": self.app_likelihood},
    #     }
