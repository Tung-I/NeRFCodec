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
    hevc_roundtrip_color, hevc_roundtrip_mono,
    av1_roundtrip_color,  av1_roundtrip_mono,
    vp9_roundtrip_color,  vp9_roundtrip_mono,
)


class PlanesCfg:
    """
    Codec-agnostic feature-plane config.
    codec: 'jpeg' | 'png' | 'hevc' | 'av1' | 'vp9'
    """

    def __init__(
        self,
        # shared
        align: int,
        codec: str = "jpeg",
        vid_pix_fmt: str = "yuv420p",  # for hevc/av1/vp9

        # density planes
        den_packing_mode: str = "flatten",
        den_quant_mode: str   = "global",
        den_global_range: tuple[float, float] = (-25.0, 25.0),
        den_r: int = 4, den_c: int = 4,
        den_quality: int | None = None,     # jpeg quality
        den_png_level: int | None = None,   # png level
        den_hevc_qp: int | None = None,     # HEVC
        den_hevc_preset: str | None = "medium",
        den_av1_qp: int | None = None,      # AV1
        den_av1_speed: int | None = 6,
        den_vp9_qp: int | None = None,      # VP9
        den_vp9_speed: int | None = 4,

        # appearance planes
        app_packing_mode: str = "flatten",
        app_quant_mode: str   = "global",
        app_global_range: tuple[float, float] = (-5.0, 5.0),
        app_r: int = 6, app_c: int = 8,
        app_quality: int | None = None,
        app_png_level: int | None = None,
        app_hevc_qp: int | None = None,
        app_hevc_preset: str | None = "medium",
        app_av1_qp: int | None = None,
        app_av1_speed: int | None = 6,
        app_vp9_qp: int | None = None,
        app_vp9_speed: int | None = 4,
    ):
        def _i(x):  return None if x is None else int(x)
        def _s(x):  return None if x is None else str(x)

        self.align = int(DCVC_ALIGN)
        self.codec = str(codec).lower()
        self.vid_pix_fmt = str(vid_pix_fmt)

        self.den_packing_mode = den_packing_mode
        self.den_quant_mode   = den_quant_mode
        self.den_global_range = den_global_range
        self.den_r, self.den_c = int(den_r), int(den_c)

        self.app_packing_mode = app_packing_mode
        self.app_quant_mode   = app_quant_mode
        self.app_global_range = app_global_range
        self.app_r, self.app_c = int(app_r), int(app_c)

        # codec params (density) — None-safe
        self.den_quality      = _i(den_quality)
        self.den_png_level    = _i(den_png_level)
        self.den_hevc_qp      = _i(den_hevc_qp)
        self.den_hevc_preset  = _s(den_hevc_preset)
        self.den_av1_qp       = _i(den_av1_qp)
        self.den_av1_speed    = _i(den_av1_speed)
        self.den_vp9_qp       = _i(den_vp9_qp)
        self.den_vp9_speed    = _i(den_vp9_speed)

        # codec params (appearance) — None-safe
        self.app_quality      = _i(app_quality)
        self.app_png_level    = _i(app_png_level)
        self.app_hevc_qp      = _i(app_hevc_qp)
        self.app_hevc_preset  = _s(app_hevc_preset)
        self.app_av1_qp       = _i(app_av1_qp)
        self.app_av1_speed    = _i(app_av1_speed)
        self.app_vp9_qp       = _i(app_vp9_qp)
        self.app_vp9_speed    = _i(app_vp9_speed)

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
    def init_ste(self, cfg: "PlanesCfg"):
        self._jpeg_cfg = cfg
        self.compression = True
        self.compress_before_volrend = True
        print("[TensorSTE] Plane codec cfg:"
            f"\n  codec={cfg.codec} align={cfg.align} pix_fmt={cfg.vid_pix_fmt}"
            f"\n  den: mode={cfg.den_packing_mode} quant={cfg.den_quant_mode} "
            f"range={cfg.den_global_range} rxc={cfg.den_r}x{cfg.den_c}"
            f"\n  app: mode={cfg.app_packing_mode} quant={cfg.app_quant_mode} "
            f"range={cfg.app_global_range} rxc={cfg.app_r}x{cfg.app_c}")

    def _codec_params_for(self, which: str):  # which in {"den","app"}
        cfg = self._jpeg_cfg
        backend = cfg.codec
        common = {"pix_fmt": cfg.vid_pix_fmt}

        def _need(val, name: str, hint: str):
            if val is None:
                raise ValueError(
                    f"[TensorSTE] Missing codec parameter '{name}' for backend='{backend}' "
                    f"(set '{which}_{hint}' in your .txt or CLI)."
                )
            return val

        if backend == "jpeg":
            q = getattr(cfg, f"{which}_quality")
            return {"quality": _need(q, "quality", "quality")}
        if backend == "png":
            lvl = getattr(cfg, f"{which}_png_level")
            return {"level": _need(lvl, "png_level", "png_level")}
        if backend == "hevc":
            qp     = _need(getattr(cfg, f"{which}_hevc_qp"),     "qp",     "hevc_qp")
            preset = getattr(cfg, f"{which}_hevc_preset") or "medium"
            return {"qp": qp, "preset": preset, **common}
        if backend == "av1":
            qp  = _need(getattr(cfg, f"{which}_av1_qp"),  "qp",  "av1_qp")
            spd = getattr(cfg, f"{which}_av1_speed")
            spd = 6 if spd is None else int(spd)
            return {"qp": qp, "cpu_used": spd, **common}
        if backend == "vp9":
            qp  = _need(getattr(cfg, f"{which}_vp9_qp"),  "qp",  "vp9_qp")
            spd = getattr(cfg, f"{which}_vp9_speed")
            spd = 4 if spd is None else int(spd)
            return {"qp": qp, "cpu_used": spd, **common}

        raise ValueError(f"Unknown codec backend: {backend}")


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
        codec: str,                 # 'jpeg' | 'png' | 'hevc' | 'av1' | 'vp9'
        codec_params: dict,         # validated per-codec kwargs from _codec_params_for
        device: torch.device,
        training: bool,
        r=4, c=4
    ):
        """
        normalize -> pack -> encode/decode -> crop -> unpack -> denorm
        Returns rec [1,C,H,W], stats {'bits','bpp'}
        """
        assert plane.dim() == 4 and plane.shape[0] == 1, "expected [1,C,H,W]"
        C, H, W = plane.shape[1:]

        # --- normalize to [0,1] for packing ---
        x = plane.to(torch.float32)
        x01, c_min, scale = normalize_planes(x, mode=quant_mode, global_range=global_range)

        # --- pack planes to a 3-channel canvas (RGB order) ---
        rgb01, (Hp, Wp) = pack_planes_to_rgb(x01, align=align, mode=packing_mode, r=r, c=c)

        use_mono = (packing_mode == "flatten")  # only mono path for 'flatten'
        codec = codec.lower()

        # --- encode/decode ---
        if use_mono:
            mono01_np = rgb01[0, 0].contiguous().cpu().numpy()  # HxW
            if   codec == "jpeg":
                rec_np, bits = jpeg_roundtrip_mono(mono01_np, quality=int(codec_params["quality"]))
            elif codec == "png":
                rec_np, bits = png_roundtrip_mono(mono01_np, level=int(codec_params["level"]))
            elif codec == "hevc":
                rec_np, bits = hevc_roundtrip_mono(
                    mono01_np,
                    qp=int(codec_params["qp"]),
                    preset=str(codec_params.get("preset", "medium")),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            elif codec == "av1":
                rec_np, bits = av1_roundtrip_mono(
                    mono01_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            elif codec == "vp9":
                rec_np, bits = vp9_roundtrip_mono(
                    mono01_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            else:
                raise ValueError(f"Unknown codec '{codec}'")

            # ensure decoded mono is 2D
            if rec_np.ndim == 3:
                rec_np = rec_np[..., 0]

            mono_rec = torch.from_numpy(rec_np).to(torch.float32)[None, None, ...]  # [1,1,Hp,Wp]
            rgb01_rec = mono_rec.repeat(1, 3, 1, 1).to(device, non_blocking=True)  # [1,3,Hp,Wp]

        else:
            rgb_np = rgb01[0].permute(1, 2, 0).contiguous().cpu().numpy()  # HxWx3 RGB
            bgr_np = np.ascontiguousarray(rgb_np[..., ::-1])               # -> BGR for OpenCV/ffmpeg
            if   codec == "jpeg":
                rec_bgr01, bits = jpeg_roundtrip_color(bgr_np, quality=int(codec_params["quality"]))
            elif codec == "png":
                rec_bgr01, bits = png_roundtrip_color(bgr_np, level=int(codec_params["level"]))
            elif codec == "hevc":
                rec_bgr01, bits = hevc_roundtrip_color(
                    bgr_np,
                    qp=int(codec_params["qp"]),
                    preset=str(codec_params.get("preset", "medium")),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            elif codec == "av1":
                rec_bgr01, bits = av1_roundtrip_color(
                    bgr_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            elif codec == "vp9":
                rec_bgr01, bits = vp9_roundtrip_color(
                    bgr_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv420p")),
                )
            else:
                raise ValueError(f"Unknown codec '{codec}'")

            rec_rgb01 = np.ascontiguousarray(rec_bgr01[..., ::-1])  # BGR->RGB
            rgb01_rec = torch.from_numpy(rec_rgb01).permute(2, 0, 1)[None, ...].to(
                device, dtype=torch.float32, non_blocking=True
            )  # [1,3,Hp,Wp]

        # --- crop away alignment padding, unpack, de-normalize ---
        rgb01_rec = crop_from_align(rgb01_rec, (Hp, Wp))
        rec01 = unpack_rgb_to_planes(rgb01_rec, C, (Hp, Wp), mode=packing_mode, r=r, c=c)
        rec = (rec01 * scale + c_min).to(torch.float32)

        # --- stats ---
        bpp = float(bits) / float(Hp * Wp)
        stats = {"bits": int(bits), "bpp": bpp, "codec": codec, "mono": bool(use_mono)}

        # --- STE ---
        if training and self._ste_enabled:
            rec = self._apply_ste(x, rec)

        return rec, stats




    # ------------------------------------------------------------------------
    # Override the "external codec" entry point to use our JPEG round-trip
    # ------------------------------------------------------------------------
    def compress_with_external_codec(self, den_feat_codec=None, app_feat_codec=None, mode: str = "train"):
        assert self._jpeg_cfg is not None, "Call init_ste(PlanesCfg(...)) first."
        training = (mode == "train")
        cfg = self._jpeg_cfg

        # ---- density ----
        self.den_rec_plane, self.den_likelihood = [], []
        den_params = self._codec_params_for("den")
        for p in self.density_plane:
            C = p.shape[1]
            if cfg.den_packing_mode == "flatten":
                assert cfg.den_r * cfg.den_c == C, f"den r*c ({cfg.den_r}*{cfg.den_c}) != C ({C})"
            rec, stats = self._im_roundtrip_plane_tensor(
                plane=p.detach(),
                quant_mode=cfg.den_quant_mode,
                global_range=cfg.den_global_range,
                packing_mode=cfg.den_packing_mode,
                align=cfg.align,
                codec=cfg.codec,
                codec_params=den_params,
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
        app_params = self._codec_params_for("app")
        for p in self.app_plane:
            C = p.shape[1]
            if cfg.app_packing_mode == "flatten":
                assert cfg.app_r * cfg.app_c == C, f"app r*c ({cfg.app_r}*{cfg.app_c}) != C ({C})"
            rec, stats = self._im_roundtrip_plane_tensor(
                plane=p.detach(),
                quant_mode=cfg.app_quant_mode,
                global_range=cfg.app_global_range,
                packing_mode=cfg.app_packing_mode,
                align=cfg.align,
                codec=cfg.codec,
                codec_params=app_params,
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

