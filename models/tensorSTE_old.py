import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2

from .tensoRF import TensorVMSplit
from .recon_utils import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    pad_to_align, crop_from_align,
    normalize_planes, DCVC_ALIGN,
    jpeg_roundtrip_color, jpeg_roundtrip_mono,
    png_roundtrip_mono, png_roundtrip_color,
)



# -------------------------------------------
# Small cfg container (optional convenience)
# -------------------------------------------
# models/tensorSTE.py  (replace the old JPEGPlanesCfg)
class JPEGPlanesCfg:
    """
    Separate configs for density and appearance planes.
    Now supports codec selection: 'jpeg' or 'png'.
    """
    def __init__(
        self,
        # shared
        align: int = DCVC_ALIGN,
        codec: str = "jpeg",  # 'jpeg' | 'png'

        # density
        den_packing_mode: str = "flatten",
        den_quant_mode: str   = "global",
        den_global_range: tuple[float, float] = (-25.0, 25.0),
        den_quality: int = 80,                 # used if codec='jpeg'
        den_png_level: int = 6,               # used if codec='png'
        den_r: int = 4, den_c: int = 4,

        # appearance
        app_packing_mode: str = "flatten",
        app_quant_mode: str   = "global",
        app_global_range: tuple[float, float] = (-5.0, 5.0),
        app_quality: int = 80,                 # jpeg
        app_png_level: int = 6,               # png
        app_r: int = 6, app_c: int = 8,
    ):
        self.align = int(align)
        self.codec = str(codec).lower()

        self.den_packing_mode = den_packing_mode
        self.den_quant_mode   = den_quant_mode
        self.den_global_range = den_global_range
        self.den_quality      = int(den_quality)
        self.den_png_level    = int(den_png_level)
        self.den_r, self.den_c = int(den_r), int(den_c)

        self.app_packing_mode = app_packing_mode
        self.app_quant_mode   = app_quant_mode
        self.app_global_range = app_global_range
        self.app_quality      = int(app_quality)
        self.app_png_level    = int(app_png_level)
        self.app_r, self.app_c = int(app_r), int(app_c)



# -------------------------------------------
# TensorSTE: TensoRF + STE+JPEG plane codec
# -------------------------------------------
class TensorSTE(TensorVMSplit):
    """
    Same triplane field & renderer as TensorVMSplit.
    Replaces adaptor feature-codec with a JPEG round-trip + STE on planes.

    How to use in your trainer:
      - build TensorSTE with the usual TensorBase/TensorVMSplit kwargs
      - call `model.init_ste(cfg_jpeg)` once (or pass via kargs and call inside __init__)
      - set `model.compression=True`, `model.compress_before_volrend=True`
      - per iteration (or when you need), call:
            model.compress_with_external_codec(mode="train" or "eval")
        This fills `self.den_rec_plane` / `self.app_rec_plane` for volume rendering.
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        # flags: keep trainer-compatible defaults
        self.using_external_codec = True   # we will ignore external codec objects and use JPEG
        self.compression = False           # will be set True after init_ste()
        self.compress_before_volrend = False
        # JPEG/STE config placeholders
        self._ste_enabled = True
        self._jpeg_cfg = None

    # ------------------------
    # Public init for JPEG cfg
    # ------------------------
    def init_ste(self, jpeg_cfg: JPEGPlanesCfg):
        self._jpeg_cfg = jpeg_cfg
        self.compression = True
        self.compress_before_volrend = True
        print("[TensorSTE] Plane codec cfg:"
            f"\n  codec={jpeg_cfg.codec} align={jpeg_cfg.align}"
            f"\n  den: mode={jpeg_cfg.den_packing_mode} quant={jpeg_cfg.den_quant_mode} "
            f"range={jpeg_cfg.den_global_range} rxc={jpeg_cfg.den_r}x{jpeg_cfg.den_c} "
            f"Q={jpeg_cfg.den_quality} L_png={jpeg_cfg.den_png_level}"
            f"\n  app: mode={jpeg_cfg.app_packing_mode} quant={jpeg_cfg.app_quant_mode} "
            f"range={jpeg_cfg.app_global_range} rxc={jpeg_cfg.app_r}x{jpeg_cfg.app_c} "
            f"Q={jpeg_cfg.app_quality} L_png={jpeg_cfg.app_png_level}")

    # ------------------------
    # Aux loss is zero for JPEG
    # ------------------------
    def get_aux_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    # ------------------------
    # Core STE helper
    # ------------------------
    @staticmethod
    def _apply_ste(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator:
          forward  -> returns 'recon'
          backward -> gradient flows to 'orig' (identity)
        """
        return recon + (orig - orig.detach())

    # ---------------------------------------------
    # JPEG round-trip for one plane tensor [1,C,H,W]
    # ---------------------------------------------
    def _im_roundtrip_plane_tensor(
        self,
        plane: torch.Tensor,
        quant_mode: str,
        global_range,
        packing_mode: str,
        align: int,
        codec: str,                 # 'jpeg' | 'png'
        quality_or_level: int,      # jpeg quality or png level
        device: torch.device,
        training: bool,
        r=4, c=4
    ):
        """
        normalize -> pack -> (mono/color) encode -> decode -> crop -> unpack -> denorm
        Returns rec [1,C,H,W], stats {'bits','bpp'}
        """
        assert plane.dim() == 4 and plane.shape[0] == 1, "expected [1,C,H,W]"
        C, H, W = plane.shape[1:]

        x = plane.to(torch.float32)
        x01, c_min, scale = normalize_planes(x, mode=quant_mode, global_range=global_range)

        # Pack to [1,3,Hp,Wp] (for 'flatten' this is 3 identical channels)
        rgb01, (Hp, Wp) = pack_planes_to_rgb(x01, align=align, mode=packing_mode, r=r, c=c)

        # === ENCODE/DECODE ===
        codec = codec.lower()
        use_mono = (packing_mode == "flatten")  # ONLY use mono for 'flatten' as requested

        if use_mono:
            # take first channel (mono), HxW float01 -> numpy
            mono01 = rgb01[:, :1]                              # [1,1,Hp,Wp]
            mono01_np = mono01[0, 0].contiguous().cpu().numpy()  # HxW

            if codec == "jpeg":
                mono_rec_np, bits = jpeg_roundtrip_mono(mono01_np, quality=int(quality_or_level))
            elif codec == "png":
                mono_rec_np, bits = png_roundtrip_mono(mono01_np, level=int(quality_or_level))
            else:
                raise ValueError(f"Unknown codec '{codec}'")

            # back to [1,3,Hp,Wp] by repeating the mono
            mono_rec = torch.from_numpy(mono_rec_np).to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            rgb01_rec = mono_rec.repeat(1, 3, 1, 1).to(device, non_blocking=True)                 # [1,3,H,W]
        else:
            # color path (used if you ever switch packing_mode != 'flatten')
            rgb_np = rgb01[0].permute(1, 2, 0).contiguous().cpu().numpy()  # HxWx3 RGB
            bgr_np = np.ascontiguousarray(rgb_np[..., ::-1])

            if codec == "jpeg":
                rec_bgr01, bits = jpeg_roundtrip_color(bgr_np, quality=int(quality_or_level))
            elif codec == "png":
                rec_bgr01, bits = png_roundtrip_color(bgr_np, level=int(quality_or_level))
            else:
                raise ValueError(f"Unknown codec '{codec}'")

            rec_rgb01 = np.ascontiguousarray(rec_bgr01[..., ::-1])
            rgb01_rec = torch.from_numpy(rec_rgb01).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32, non_blocking=True)

        # crop (no-op if already aligned)
        rgb01_rec = crop_from_align(rgb01_rec, (Hp, Wp))

        # Unpack back to planes in [0,1]
        rec01 = unpack_rgb_to_planes(rgb01_rec, C, (Hp, Wp), mode=packing_mode, r=r, c=c)

        # Denorm to raw
        rec = (rec01 * scale + c_min).to(torch.float32)

        # Stats
        bpp = float(bits) / float(Hp * Wp)
        stats = {"bits": int(bits), "bpp": bpp, "codec": codec, "mono": bool(use_mono)}

        if training and self._ste_enabled:
            rec = self._apply_ste(x, rec)

        return rec, stats


    # ------------------------------------------------------------------------
    # Override the "external codec" entry point to use our JPEG round-trip
    # ------------------------------------------------------------------------
    def compress_with_external_codec(self, den_feat_codec=None, app_feat_codec=None, mode: str = "train"):
        assert self._jpeg_cfg is not None, "Call init_ste(JPEGPlanesCfg(...)) first."
        training = (mode == "train")
        cfg = self._jpeg_cfg

        # ---- density ----
        self.den_rec_plane, self.den_likelihood = [], []
        for p in self.density_plane:
            C = p.shape[1]
            if cfg.den_packing_mode == "flatten":
                assert cfg.den_r * cfg.den_c == C, f"den r*c ({cfg.den_r}*{cfg.den_c}) != C ({C})"
            q_or_l = cfg.den_quality if cfg.codec == "jpeg" else cfg.den_png_level
            rec, stats = self._im_roundtrip_plane_tensor(
                plane=p.detach(),
                quant_mode=cfg.den_quant_mode,
                global_range=cfg.den_global_range,
                packing_mode=cfg.den_packing_mode,
                align=cfg.align,
                codec=cfg.codec,
                quality_or_level=q_or_l,
                device=self.device,
                training=training,
                r=cfg.den_r, c=cfg.den_c,
            )
            if training and self._ste_enabled:
                rec = self._apply_ste(p, rec)
            self.den_rec_plane.append(rec)
            self.den_likelihood.append(stats)

        # ---- appearance ----
        self.app_rec_plane, self.app_likelihood = [], []
        for p in self.app_plane:
            C = p.shape[1]
            if cfg.app_packing_mode == "flatten":
                assert cfg.app_r * cfg.app_c == C, f"app r*c ({cfg.app_r}*{cfg.app_c}) != C ({C})"
            q_or_l = cfg.app_quality if cfg.codec == "jpeg" else cfg.app_png_level
            rec, stats = self._im_roundtrip_plane_tensor(
                plane=p.detach(),
                quant_mode=cfg.app_quant_mode,
                global_range=cfg.app_global_range,
                packing_mode=cfg.app_packing_mode,
                align=cfg.align,
                codec=cfg.codec,
                quality_or_level=q_or_l,
                device=self.device,
                training=training,
                r=cfg.app_r, c=cfg.app_c,
            )
            if training and self._ste_enabled:
                rec = self._apply_ste(p, rec)
            self.app_rec_plane.append(rec)
            self.app_likelihood.append(stats)

        return {
            "den": {"rec_planes": self.den_rec_plane, "rec_likelihood": self.den_likelihood},
            "app": {"rec_planes": self.app_rec_plane, "rec_likelihood": self.app_likelihood},
        }


    # --------------------------------------------------------
    # Optional toggles you may call from the training script
    # --------------------------------------------------------
    def set_ste(self, enabled: bool = True):
        self._ste_enabled = bool(enabled)
        print(f"[TensorSTE] STE {'enabled' if self._ste_enabled else 'disabled'}")

    def set_compress_before_volrend(self, enabled: bool = True):
        self.compress_before_volrend = bool(enabled)
        print(f"[TensorSTE] compress_before_volrend = {self.compress_before_volrend}")
