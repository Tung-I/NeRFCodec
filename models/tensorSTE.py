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

class _SPSASurrogateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        orig, rec_detached,
        codec_fn,
        eps: float,
        n_samples: int,
        block_size: int,
        share_delta_across_channels: bool,
        include_ste_grad: bool,
        spsa_lambda: float,
        normalize_proj: bool,
    ):
        """
        orig:         [1,C,H,W] tensor requiring grad
        rec_detached: [1,C,H,W] decoded plane (detached) used for forward
        codec_fn:     python callable: Tensor -> Tensor (no grad)
        """
        ctx.codec_fn = codec_fn
        ctx.eps = float(eps)
        ctx.n_samples = int(n_samples)
        ctx.block_size = int(block_size)
        ctx.share_delta_across_channels = bool(share_delta_across_channels)
        ctx.include_ste_grad = bool(include_ste_grad)
        ctx.spsa_lambda = float(spsa_lambda)
        ctx.normalize_proj = bool(normalize_proj)
        ctx.save_for_backward(orig)
        return rec_detached

    @staticmethod
    def backward(ctx, grad_out):
        (orig,) = ctx.saved_tensors
        codec_fn = ctx.codec_fn
        eps = ctx.eps
        n = max(1, ctx.n_samples)
        B = max(1, ctx.block_size)
        shareC = ctx.share_delta_across_channels
        include_ste = ctx.include_ste_grad
        lam = ctx.spsa_lambda
        normalize_proj = ctx.normalize_proj

        # Hybrid gradient:
        #   g_total = (STE identity term) + lam * (SPSA term)
        # where SPSA term estimates J^T grad_out.
        with torch.no_grad():
            x0 = orig.detach()
            _, C, H, W = x0.shape

            g_spsa = torch.zeros_like(x0)

            # block grid
            Hb = (H + B - 1) // B
            Wb = (W + B - 1) // B

            for _ in range(n):
                # ---- blockwise Rademacher delta ----
                if shareC:
                    # one delta shared across channels -> lower variance
                    delta_small = torch.empty((1, 1, Hb, Wb), device=x0.device, dtype=x0.dtype)
                    delta_small.bernoulli_(0.5).mul_(2.0).sub_(1.0)  # ±1
                    delta_small = delta_small.expand(1, C, Hb, Wb)
                else:
                    delta_small = torch.empty((1, C, Hb, Wb), device=x0.device, dtype=x0.dtype)
                    delta_small.bernoulli_(0.5).mul_(2.0).sub_(1.0)  # ±1

                # upsample by repeating blocks, then crop to exact H,W
                delta = delta_small.repeat_interleave(B, dim=2).repeat_interleave(B, dim=3)
                delta = delta[:, :, :H, :W]

                y_plus  = codec_fn(x0 + eps * delta)
                y_minus = codec_fn(x0 - eps * delta)

                diff = (y_plus - y_minus)

                # proj = <grad_out, diff> / (2 eps)
                proj = (grad_out * diff).sum() / (2.0 * eps)

                # Optional explicit normalization (keeps scale stable across resolutions)
                if normalize_proj:
                    proj = proj / grad_out.numel()

                g_spsa += proj * delta

            g_spsa /= float(n)

            # identity term (STE): dL/dx += grad_out
            g_total = (grad_out if include_ste else 0.0) + (lam * g_spsa)

        # Only orig gets gradients; others are None
        return g_total, None, None, None, None, None, None, None, None, None, None



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

         # NEW: gradient surrogate selection
        grad_surrogate_mode: str = "ste",   # "ste" | "mste_std"
        grad_surrogate_std_eps: float = 1e-8,
        spsa_n_samples: int = 1,
        spsa_gate_on_cache_refresh: int = 1,
        # SPSA variance reduction + hybrid
        spsa_block_size: int = 16,
        spsa_share_delta_across_channels: int = 1,  # bool-ish
        spsa_hybrid_lambda: float = 0.01,            # λ in STE + λ*SPSA
        spsa_normalize_proj: int = 1,                # keep your /numel behavior

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

        self.grad_surrogate_mode = str(grad_surrogate_mode)
        self.grad_surrogate_std_eps = float(grad_surrogate_std_eps) 
        def _default_eps_from_range(rng):
            lo, hi = rng
            return float(hi - lo) / 255.0  # 1 LSB in normalized space
        
        self.spsa_gate_on_cache_refresh = spsa_gate_on_cache_refresh
        # self.den_spsa_eps = float(den_spsa_eps) if den_spsa_eps is not None else _default_eps_from_range(self.den_global_range)
        # self.app_spsa_eps = float(app_spsa_eps) if app_spsa_eps is not None else _default_eps_from_range(self.app_global_range)
        self.den_spsa_eps = _default_eps_from_range(self.den_global_range) 
        self.app_spsa_eps = _default_eps_from_range(self.app_global_range) 
        self.spsa_n_samples = int(spsa_n_samples)

        self.spsa_block_size = int(spsa_block_size)
        self.spsa_share_delta_across_channels = bool(spsa_share_delta_across_channels)
        self.spsa_hybrid_lambda = float(spsa_hybrid_lambda)
        self.spsa_normalize_proj = bool(spsa_normalize_proj)

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
        self.using_external_codec = True
        self.compression = False
        self.compress_before_volrend = False
        self._ste_enabled = True
        self._jpeg_cfg = None

        # -------- Codec cache (single-frame, per-plane) --------
        self._cache_refresh_k   = 1      # run codec every K calls
        self._cache_refresh_eps = 0.0    # relative L2 threshold to force refresh
        self._cache_bpp_refresh_k = 1    # (kept for parity; used when you add grad-bpp)
        self._cache_last_step   = -10**9
        self._cache_ticks       = 0
        self._cache_refreshed_this_call = False

        # density caches (lists per-plane index)
        self._den_cache_rec    = None    # list[Tensor] of [1,C,H,W] (detached, on self.device)
        self._den_cache_stats  = None    # list[dict]
        self._den_cache_snap   = None    # list[Tensor] snapshots of raw planes (detached)

        # appearance caches
        self._app_cache_rec    = None
        self._app_cache_stats  = None
        self._app_cache_snap   = None

    # ------------------------
    # Public init for JPEG cfg
    # ------------------------
    def init_ste(self, cfg: "PlanesCfg"):
        self._jpeg_cfg = cfg
        self.compression = True
        self.compress_before_volrend = True
        # ---- cache knobs (optional; default=old behavior) ----
        self._cache_refresh_k     = int(getattr(cfg, "codec_refresh_k", 1))
        self._cache_refresh_eps   = float(getattr(cfg, "refresh_trigger_eps", 0.0))
        self._cache_bpp_refresh_k = int(getattr(cfg, "bpp_refresh_k", 1))
        print("[TensorSTE] Plane codec cfg:"
              f"\n  codec={cfg.codec} align={cfg.align} pix_fmt={cfg.vid_pix_fmt}"
              f"\n  grad_surrogate: mode={getattr(cfg,'grad_surrogate_mode',None)} std_eps={getattr(cfg,'grad_surrogate_std_eps',None)}"
              f"\n  den: mode={cfg.den_packing_mode} quant={cfg.den_quant_mode} "
              f"range={cfg.den_global_range} rxc={cfg.den_r}x{cfg.den_c}"
              f"\n  app: mode={cfg.app_packing_mode} quant={cfg.app_quant_mode} "
              f"range={cfg.app_global_range} rxc={cfg.app_r}x{cfg.app_c}"
              f"\n  cache: K={self._cache_refresh_k} eps={self._cache_refresh_eps} "
              f"bppK={self._cache_bpp_refresh_k}")

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
    # Cache refresh routine
    # ------------------------

    @torch.no_grad()
    def _rel_change(self, cur: torch.Tensor, snap: torch.Tensor) -> float:
        if snap is None or cur.shape != snap.shape:
            return float("inf")
        num = (cur - snap).float().pow(2).sum()
        den = snap.float().pow(2).sum().clamp_min(1e-12)
        return float((num / den).sqrt().item())

    @torch.no_grad()
    def _planes_changed(self, curr_list: list[torch.Tensor], snap_list: list[torch.Tensor] | None) -> bool:
        if self._cache_refresh_eps <= 0.0:
            return False
        if snap_list is None or len(snap_list) != len(curr_list):
            return True
        for cur, snap in zip(curr_list, snap_list):
            if self._rel_change(cur, snap) > self._cache_refresh_eps:
                print(f"[TensorSTE] Plane changed: {self._rel_change(cur, snap)} > {self._cache_refresh_eps}")
                return True
        return False

    def _gather_den_planes(self) -> list[torch.Tensor]:
        # self.density_plane is a list of parameters shaped [1,C,H,W]
        return [p for p in self.density_plane]

    def _gather_app_planes(self) -> list[torch.Tensor]:
        return [p for p in self.app_plane]

    def set_codec_cache(self, refresh_k: int | None = None, refresh_eps: float | None = None, bpp_refresh_k: int | None = None):
        if refresh_k is not None:
            self._cache_refresh_k = int(refresh_k)
        if refresh_eps is not None:
            self._cache_refresh_eps = float(refresh_eps)
        if bpp_refresh_k is not None:
            self._cache_bpp_refresh_k = int(bpp_refresh_k)
        print(f"[TensorSTE] cache knobs -> K={self._cache_refresh_k}, eps={self._cache_refresh_eps}, bppK={self._cache_bpp_refresh_k}")

    @torch.no_grad()
    def _maybe_refresh_codec_cache(self, training: bool):
        """Refresh cached (detached) reconstructions and stats if K steps passed or planes changed a lot."""
        self._cache_refreshed_this_call = False
        self._cache_ticks += 1
        do_time = (self._cache_refresh_k <= 1) or (self._cache_ticks - self._cache_last_step >= self._cache_refresh_k)

        den_now = self._gather_den_planes()
        app_now = self._gather_app_planes()
        do_change = (self._planes_changed(den_now, self._den_cache_snap) or
                     self._planes_changed(app_now, self._app_cache_snap))

        if not (do_time or do_change):
            return  # still fresh

        self._cache_refreshed_this_call = True
        cfg = self._jpeg_cfg
        # --- prepare per-which codec params once ---
        den_params = self._codec_params_for("den")
        app_params = self._codec_params_for("app")

        # --- recompute density cache ---
        self._den_cache_rec, self._den_cache_stats, self._den_cache_snap = [], [], []
        for p in den_now:
            C = p.shape[1]
            if cfg.den_packing_mode == "flatten":
                assert cfg.den_r * cfg.den_c == C, f"den r*c ({cfg.den_r}*{cfg.den_c}) != C ({C})"
            rec, stats = self._im_roundtrip_plane_tensor(
                plane=p.detach(),  # encode current raw
                quant_mode=cfg.den_quant_mode,
                global_range=cfg.den_global_range,
                packing_mode=cfg.den_packing_mode,
                align=cfg.align,
                codec=cfg.codec,
                codec_params=den_params,
                device=self.device,
                training=False,    # <- no STE here; applied later on-the-fly
                r=cfg.den_r, c=cfg.den_c,
            )
            self._den_cache_rec.append(rec.detach())
            self._den_cache_stats.append(stats)
            self._den_cache_snap.append(p.detach().clone())

        # --- recompute appearance cache ---
        self._app_cache_rec, self._app_cache_stats, self._app_cache_snap = [], [], []
        for p in app_now:
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
                training=False,   # <- no STE here
                r=cfg.app_r, c=cfg.app_c,
            )
            self._app_cache_rec.append(rec.detach())
            self._app_cache_stats.append(stats)
            self._app_cache_snap.append(p.detach().clone())

        self._cache_last_step = self._cache_ticks

    # ------------------------
    # Aux loss is zero for JPEG
    # ------------------------
    def get_aux_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    # ------------------------
    # Core STE helper
    # ------------------------
    # @staticmethod
    # def _apply_ste(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    #     """
    #     Straight-through estimator:
    #       forward  -> returns 'recon'
    #       backward -> gradient flows to 'orig' (identity)
    #     """
    #     return recon + (orig - orig.detach())

    @staticmethod
    def _apply_ste(
        orig: torch.Tensor,
        recon: torch.Tensor,
        mode: str = "ste",
        std_eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Gradient surrogate for non-differentiable codec / quantization.

        mode="ste":
            y = recon + (orig - orig.detach())
            Forward: y == recon
            Backward: dy/dorig = I

        mode="mste_std":
            Implements mSTE from Mack et al. (and used in SCALED):
              y = orig + sg(eps) * ( sigma(eps) / sg(sigma(eps)) )
            where eps = recon - orig, sigma(.) is std over eps.
            Forward: sigma/sigma_detach == 1  => y == orig + sg(eps) == recon
            Backward: additional stabilizing term flows through sigma(eps).
        """
        mode = (mode or "ste").lower()

        if mode in ("ste", "identity"):
            return recon + (orig - orig.detach())

        if mode in ("mste_std", "mste"):
            eps = recon - orig                         # Qe
            sigma = torch.std(eps, unbiased=False)     # σ_Qe (scalar)
            # avoid divide-by-zero / NaNs; keep ratio ~1 in forward
            sigma_safe = torch.clamp(sigma, min=std_eps)
            sigma_det  = torch.clamp(sigma.detach(), min=std_eps)

            # orig + sg[eps] * (sigma / sg[sigma])
            return orig + eps.detach() * (sigma_safe / sigma_det)

        raise ValueError(f"Unknown grad surrogate mode: {mode}")

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
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
                )
            elif codec == "av1":
                rec_np, bits = av1_roundtrip_mono(
                    mono01_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
                )
            elif codec == "vp9":
                rec_np, bits = vp9_roundtrip_mono(
                    mono01_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
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
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
                )
            elif codec == "av1":
                rec_bgr01, bits = av1_roundtrip_color(
                    bgr_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
                )
            elif codec == "vp9":
                rec_bgr01, bits = vp9_roundtrip_color(
                    bgr_np,
                    qp=int(codec_params["qp"]),
                    cpu_used=int(codec_params["cpu_used"]),
                    pix_fmt=str(codec_params.get("pix_fmt", "yuv444p")),
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
            # rec = self._apply_ste(x, rec)
            cfg = self._jpeg_cfg
            rec = self._apply_ste(
                x, rec,
                mode=getattr(cfg, "grad_surrogate_mode", "ste"),
                std_eps=float(getattr(cfg, "grad_surrogate_std_eps", 1e-8)),
            )

        return rec, stats


    # ------------------------------------------------------------------------
    # Override the "external codec" entry point to use our JPEG round-trip
    # ------------------------------------------------------------------------
    def compress_with_external_codec(self, den_feat_codec=None, app_feat_codec=None, mode: str = "train"):
        assert self._jpeg_cfg is not None, "Call init_ste(PlanesCfg(...)) first."
        training = (mode == "train")

        # 1) refresh cache if needed (slow path only every K or when changed)
        self._maybe_refresh_codec_cache(training=training)

        # 2) serve cached recs; apply STE on-the-fly if training
        self.den_rec_plane, self.den_likelihood = [], []
        self.app_rec_plane, self.app_likelihood = [], []

        # density
        if self._den_cache_rec is None:
            # if user calls before first refresh and K>1, force a refresh now
            self._maybe_refresh_codec_cache(training=training)
        for p, rec_detached, stats in zip(self.density_plane, self._den_cache_rec, self._den_cache_stats):
            # rec = rec_detached
            # if training and self._ste_enabled:
            #     # rec = self._apply_ste(p, rec_detached)  # STE: forward=rec, grad=identity wrt p
            #     cfg = self._jpeg_cfg
            #     rec = self._apply_ste(
            #         p, rec_detached,
            #         mode=getattr(cfg, "grad_surrogate_mode", "ste"),
            #         std_eps=float(getattr(cfg, "grad_surrogate_std_eps", 1e-8)),
            #     )

            # self.den_rec_plane.append(rec)
            # self.den_likelihood.append(stats)
            rec = rec_detached
            if training and self._ste_enabled:
                rec = self._apply_grad_surrogate(p, rec_detached, which="den")
            self.den_rec_plane.append(rec)
            self.den_likelihood.append(stats)

        # appearance
        if self._app_cache_rec is None:
            self._maybe_refresh_codec_cache(training=training)
        for p, rec_detached, stats in zip(self.app_plane, self._app_cache_rec, self._app_cache_stats):
            # rec = rec_detached
            # if training and self._ste_enabled:
            #     # rec = self._apply_ste(p, rec_detached)
            #     cfg = self._jpeg_cfg
            #     rec = self._apply_ste(
            #         p, rec_detached,
            #         mode=getattr(cfg, "grad_surrogate_mode", "ste"),
            #         std_eps=float(getattr(cfg, "grad_surrogate_std_eps", 1e-8)),
            #     )

            # self.app_rec_plane.append(rec)
            # self.app_likelihood.append(stats)
            rec = rec_detached
            if training and self._ste_enabled:
                rec = self._apply_grad_surrogate(p, rec_detached, which="app")
            self.app_rec_plane.append(rec)
            self.app_likelihood.append(stats)

        return {
            "den": {"rec_planes": self.den_rec_plane, "rec_likelihood": self.den_likelihood},
            "app": {"rec_planes": self.app_rec_plane, "rec_likelihood": self.app_likelihood},
        }


    def set_ste(self, enabled: bool = True):
        self._ste_enabled = bool(enabled)
        print(f"[TensorSTE] STE {'enabled' if self._ste_enabled else 'disabled'}")

    def set_compress_before_volrend(self, enabled: bool = True):
        self.compress_before_volrend = bool(enabled)
        print(f"[TensorSTE] compress_before_volrend = {self.compress_before_volrend}")


    def _roundtrip_plane_no_surrogate(self, plane: torch.Tensor, which: str) -> torch.Tensor:
        """
        which: "den" or "app"
        Returns: decoded plane [1,C,H,W] on self.device, no STE/mSTE/SPSA applied.
        """
        cfg = self._jpeg_cfg
        assert cfg is not None

        if which == "den":
            quant_mode = cfg.den_quant_mode
            global_range = cfg.den_global_range
            packing_mode = cfg.den_packing_mode
            r, c = cfg.den_r, cfg.den_c
            codec_params = self._codec_params_for("den")
        elif which == "app":
            quant_mode = cfg.app_quant_mode
            global_range = cfg.app_global_range
            packing_mode = cfg.app_packing_mode
            r, c = cfg.app_r, cfg.app_c
            codec_params = self._codec_params_for("app")
        else:
            raise ValueError(which)

        rec, _stats = self._im_roundtrip_plane_tensor(
            plane=plane,
            quant_mode=quant_mode,
            global_range=global_range,
            packing_mode=packing_mode,
            align=cfg.align,
            codec=cfg.codec,
            codec_params=codec_params,
            device=self.device,
            training=False,   # IMPORTANT: no surrogate here
            r=r, c=c
        )
        return rec
    
    def _apply_grad_surrogate(self, orig: torch.Tensor, rec_detached: torch.Tensor, which: str) -> torch.Tensor:
        cfg = self._jpeg_cfg
        mode = getattr(cfg, "grad_surrogate_mode", "ste").lower()

        # default eps for mSTE
        std_eps = float(getattr(cfg, "grad_surrogate_std_eps", 1e-8))

        if mode in ("ste", "identity"):
            return self._apply_ste(orig, rec_detached, mode="ste", std_eps=std_eps)

        if mode in ("mste_std", "mste"):
            return self._apply_ste(orig, rec_detached, mode="mste_std", std_eps=std_eps)

        if mode in ("spsa", "hybrid_spsa"):
            # speed hack: only do SPSA when the cache actually refreshed
            gate = bool(getattr(cfg, "spsa_gate_on_cache_refresh", True))
            if gate and (not self._cache_refreshed_this_call):
                return self._apply_ste(orig, rec_detached, mode="ste", std_eps=std_eps)

            eps = cfg.den_spsa_eps if which == "den" else cfg.app_spsa_eps
            n_samples = int(getattr(cfg, "spsa_n_samples", 1))

            block_size = int(getattr(cfg, "spsa_block_size", 16))
            shareC = bool(getattr(cfg, "spsa_share_delta_across_channels", True))
            normalize_proj = bool(getattr(cfg, "spsa_normalize_proj", True))

            # hybrid controls
            include_ste_grad = (mode == "hybrid_spsa")
            lam = float(getattr(cfg, "spsa_hybrid_lambda", 0.01)) if include_ste_grad else 1.0

            def codec_fn(x):
                return self._roundtrip_plane_no_surrogate(x, which)

            return _SPSASurrogateFn.apply(
                orig, rec_detached,
                codec_fn,
                eps,
                n_samples,
                block_size,
                shareC,
                include_ste_grad,
                lam,
                normalize_proj,
            )


        raise ValueError(f"Unknown grad surrogate mode: {mode}")



