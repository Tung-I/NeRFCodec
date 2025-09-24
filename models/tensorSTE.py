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
    jpeg_roundtrip_color,  # CPU JPEG round-trip for [1,3,H,W] float01 RGB
)


# -------------------------------------------
# Small cfg container (optional convenience)
# -------------------------------------------
class JPEGPlanesCfg:
    def __init__(
        self,
        plane_packing_mode: str = "tile",   # your previous "plane_packing_mode"
        quant_mode: str = "global",         # "global" recommended
        global_range: tuple[float, float] = (-20.0, 20.0),
        align: int = DCVC_ALIGN,            # keep 64-alignment by default
        quality: int = 80,                  # JPEG quality
    ):
        self.plane_packing_mode = plane_packing_mode
        self.quant_mode = quant_mode
        self.global_range = global_range
        self.align = int(align)
        self.quality = int(quality)


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
        """
        Provide JPEG settings. Call once after construction.
        """
        self._jpeg_cfg = jpeg_cfg
        # Tell the main loop that we do have a compression path
        self.compression = True
        self.compress_before_volrend = True
        print(f"[TensorSTE] JPEG planes: mode={jpeg_cfg.plane_packing_mode}, "
              f"quant={jpeg_cfg.quant_mode}, range={jpeg_cfg.global_range}, "
              f"align={jpeg_cfg.align}, quality={jpeg_cfg.quality}")

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
    def _jpeg_roundtrip_plane_tensor(self, plane: torch.Tensor, quant_mode: str, global_range, packing_mode: str,
                                     align: int, quality: int, device: torch.device, training: bool, r=4, c=4):
        """
        Implements: normalize -> pack -> pad -> JPEG -> crop -> unpack -> denorm
        Returns:
            rec_plane [1,C,H,W] (same shape/device as input),
            stats dict {"bits": int, "bpp": float}
        """
        assert plane.dim() == 4 and plane.shape[0] == 1, "expected [1,C,H,W] plane"
        C, H, W = plane.shape[1:]

        # (1) Normalize to [0,1] per your chosen scheme (global recommended)
        #     normalize_planes returns x01, c_min, scale; works on [1,C,H,W]
        x = plane.to(torch.float32)
        x01, c_min, scale = normalize_planes(x, mode=quant_mode, global_range=global_range)  # all on current device

        # (2) Pack C channels into 3 RGB (your utility packs arbitrary C)
        rgb01, (Hp, Wp) = pack_planes_to_rgb(x01, align=align, mode=packing_mode, r=r, c=c)   # [1,3,Hp,Wp], float01

        # (3) JPEG CPU round-trip (expects float01 RGB [1,3,H,W])
        rgb01_cpu_rgb = rgb01[0].permute(1, 2, 0).contiguous().cpu().numpy() 

        bgr01 = np.ascontiguousarray(rgb01_cpu_rgb[..., ::-1])

        rec_bgr01, bits = jpeg_roundtrip_color(bgr01, quality=quality)

        rec_rgb01 = np.ascontiguousarray(rec_bgr01[..., ::-1])  # BGR→RGB
        rgb01_rec = torch.from_numpy(rec_rgb01).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32, non_blocking=True)  # [1,3,Hp,Wp] float01 on CPU

        rgb01_rec = crop_from_align(rgb01_rec, (Hp, Wp))  # no-op if already aligned; safe to keep
        rec01 = unpack_rgb_to_planes(rgb01_rec, C, (Hp, Wp), mode=packing_mode, r=r, c=c)

        # (5) De-normalize back to raw domain
        rec = (rec01 * scale + c_min).to(torch.float32)

        # Stats
        bpp = float(bits) / float(Hp * Wp)
        stats = {"bits": int(bits), "bpp": bpp}

        # For training we’ll return STE’d tensor;
        # for eval we just return the decoded recon
        if training and self._ste_enabled:
            rec = self._apply_ste(x, rec)

        return rec, stats

    # ------------------------------------------------------------------------
    # Override the "external codec" entry point to use our JPEG round-trip
    # ------------------------------------------------------------------------
    @torch.no_grad()  # the JPEG path is non-differentiable; STE wraps grads outside
    def compress_with_external_codec(self, den_feat_codec=None, app_feat_codec=None, mode: str = "train"):
        """
        Reconstruct density & appearance planes via JPEG round-trip + (optionally) STE.

        Returns a dict:
        {
            "den": {"rec_planes": [Pxy, Pxz, Pyz], "rec_likelihood": [dict,...]},
            "app": {"rec_planes": [Pxy, Pxz, Pyz], "rec_likelihood": [dict,...]},
        }
        where each dict at least has: {"bits": int, "bpp": float}
        """
        assert self._jpeg_cfg is not None, "Call init_ste(JPEGPlanesCfg(...)) before compression."

        training = (mode == "train")
        cfg = self._jpeg_cfg

        # ---------- (one-time) debug stats to pick global quant ranges ----------
        if getattr(self, "_debug_print_range", True):
            with torch.no_grad():
                def _stats(cat_name, planes):
                    flat = torch.cat([p.reshape(-1) for p in planes], dim=0)
                    mn, mx = flat.min().item(), flat.max().item()
                    mu, sd = flat.mean().item(), flat.std(unbiased=False).item()
                    print(f"[{cat_name}] min={mn:.6f} max={mx:.6f} mean={mu:.6f} std={sd:.6f}")
                    # Per-plane extremes (optional, uncomment if you want detail)
                    # for idx, p in enumerate(planes):
                    #     print(f"  {cat_name}[{idx}] range=({p.min().item():.6f}, {p.max().item():.6f})")

                # print("==== DEBUG plane ranges (to choose global quant range) ====")
                # _stats("density", [p.detach().cpu() for p in self.density_plane])
                # _stats("appearance", [p.detach().cpu() for p in self.app_plane])
                # print("==========================================================")

                # ==== DEBUG plane ranges (to choose global quant range) ====
                # [density] min=-24.481936 max=23.226543 mean=0.017528 std=0.965079
                # [appearance] min=-4.445703 max=4.541358 mean=-0.009613 std=0.924702

            # raise Exception

            self._debug_print_range = False  # print once

        # ---- density planes (identity map) ----
        self.den_rec_plane, self.den_likelihood = [], []
        for i in range(len(self.density_plane)):
            src_plane = self.density_plane[i]  # [1,C,H,W]
            rec, stats = self._jpeg_roundtrip_plane_tensor(
                plane=src_plane.detach(),
                quant_mode=cfg.quant_mode,
                global_range=cfg.global_range,
                packing_mode=cfg.plane_packing_mode,
                align=cfg.align,
                quality=cfg.quality,
                device=self.device,
                training=training,
                r=4, c=4,  
            )
            self.den_rec_plane.append(rec)
            self.den_likelihood.append(stats)

        # ---- appearance planes (NO tanh; rely on global normalization) ----
        self.app_rec_plane, self.app_likelihood = [], []
        for i in range(len(self.app_plane)):
            src_plane = self.app_plane[i]  # <—— NO tanh here
            rec, stats = self._jpeg_roundtrip_plane_tensor(
                plane=src_plane.detach(),
                quant_mode=cfg.quant_mode,
                global_range=cfg.global_range,
                packing_mode=cfg.plane_packing_mode,
                align=cfg.align,
                quality=cfg.quality,
                device=self.device,
                training=training,
                r=6, c=8, 
            )
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
