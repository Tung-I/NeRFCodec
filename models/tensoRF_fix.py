from .tensoRF import TensorVMSplit

# -------------------------------------------
# FixTensorVMSplit: freeze selected codec parts
# -------------------------------------------
class FixTensorVMSplit(TensorVMSplit):
    """
    Variant that can fix (freeze) either 'entropy_parameters' or 'context_prediction'
    in the neural image codecs, while keeping adaptors (and quantiles) trainable.

    Args (via __init__ kargs):
        fix_entropy_parameters: bool = False
        fix_context_prediction: bool = False
        freeze_scope: str = "both"     # {"both","den","app"}
        exclude_frozen_from_bitrate: bool = True
            If True, estimate_codec_transmission_bits() subtracts any frozen groups
            from the reported totals (treated as shared/global, not per-content).
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.fix_entropy_parameters   = bool(kargs.get("fix_entropy_parameters", False))
        self.fix_context_prediction   = bool(kargs.get("fix_context_prediction", False))
        self.freeze_scope             = str(kargs.get("freeze_scope", "both")).lower()  # "both"|"den"|"app"
        self.exclude_frozen_from_bitrate = bool(kargs.get("exclude_frozen_from_bitrate", True))

        assert self.freeze_scope in ("both", "den", "app"), "freeze_scope must be one of {'both','den','app'}"
        if self.fix_entropy_parameters or self.fix_context_prediction:
            print(f"[FixTensorVMSplit] Freeze plan: "
                  f"{'entropy_parameters ' if self.fix_entropy_parameters else ''}"
                  f"{'context_prediction' if self.fix_context_prediction else ''} "
                  f"on scope='{self.freeze_scope}'")

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _freeze_submodule(mdl, name: str):
        """Set requires_grad=False for all params in mdl.<name>, and .eval() if present."""
        if not hasattr(mdl, name):
            return False
        sub = getattr(mdl, name)
        try:
            sub.eval()
        except Exception:
            pass
        n_params = 0
        for p in sub.parameters(recurse=True):
            p.requires_grad = False
            n_params += p.numel()
        print(f"[FixTensorVMSplit] Froze '{name}' ({n_params} params).")
        return True

    def _apply_freeze_to_codec(self, codec, tag: str):
        """Freeze chosen parts on one codec (den/app)."""
        if self.fix_entropy_parameters:
            if not self._freeze_submodule(codec, "entropy_parameters"):
                print(f"[FixTensorVMSplit] WARN: '{tag}' has no 'entropy_parameters'")
        if self.fix_context_prediction:
            if not self._freeze_submodule(codec, "context_prediction"):
                print(f"[FixTensorVMSplit] WARN: '{tag}' has no 'context_prediction'")

    def _scope_includes(self, which: str) -> bool:
        if self.freeze_scope == "both":
            return True
        return self.freeze_scope == which

    # ---- codec init + targeted freezing ------------------------------------

    def init_feat_codec(
        self,
        codec_ckpt_path: str = "",
        loading_pretrain_param: bool = True,
        adaptor_q_bit: int = 8,
        codec_backbone_type: str = "cheng2020-anchor",
    ):
        super().init_feat_codec(
            codec_ckpt_path=codec_ckpt_path,
            loading_pretrain_param=loading_pretrain_param,
            adaptor_q_bit=adaptor_q_bit,
            codec_backbone_type=codec_backbone_type,
        )
        # Apply freezes right after codecs are constructed
        if self._scope_includes("den"):
            self._apply_freeze_to_codec(self.den_feat_codec, "den_feat_codec")
        if self._scope_includes("app"):
            self._apply_freeze_to_codec(self.app_feat_codec, "app_feat_codec")

    # ---- optimizer groups (inherit base behavior) ---------------------------
    # We rely on requires_grad=False to exclude frozen parts automatically.
    # Adaptors and quantiles remain trainable as usual.

    # ---- bitrate accounting that respects frozen parts ----------------------

    def estimate_codec_transmission_bits(
        self,
        mode: str = "raw",
        q_bits: int = 8,
        include_header: bool = True,
        return_breakdown: bool = True,
    ):
        """
        Start from the base computation, then (optionally) exclude the frozen groups
        from per-content transmission totals, treating them as global/shared.
        """
        out = super().estimate_codec_transmission_bits(
            mode=mode, q_bits=q_bits, include_header=include_header, return_breakdown=return_breakdown
        )
        if not self.exclude_frozen_from_bitrate:
            return out

        def _zero_group(codec_dict, group_name):
            # codec_dict is like out["density_codec"] or out["appearance_codec"]
            if "breakdown_bits" not in codec_dict:
                return 0.0, 0.0
            b_bits = codec_dict["breakdown_bits"].get(group_name, 0.0)
            b_MB   = codec_dict.get("breakdown_MB", {}).get(group_name, 0.0)
            if b_bits == 0.0:
                return 0.0, 0.0
            codec_dict["breakdown_bits"][group_name] = 0.0
            if "breakdown_MB" in codec_dict:
                codec_dict["breakdown_MB"][group_name] = 0.0
            codec_dict["total_bits"] -= b_bits
            codec_dict["total_MB"]   -= b_MB
            return b_bits, b_MB

        # Keep snapshots for clarity
        out["frozen_excluded"] = {}
        removed_bits_total = 0.0
        removed_MB_total   = 0.0

        def _maybe_exclude(side: str):
            nonlocal removed_bits_total, removed_MB_total
            if side == "den" and not self._scope_includes("den"): return
            if side == "app" and not self._scope_includes("app"): return

            key = "density_codec" if side == "den" else "appearance_codec"
            snapshot = dict(out[key])  # shallow copy for logging
            rm_bits = rm_MB = 0.0

            if self.fix_entropy_parameters:
                b, m = _zero_group(out[key], "entropy_parameters")
                rm_bits += b; rm_MB += m
            if self.fix_context_prediction:
                b, m = _zero_group(out[key], "context_prediction")
                rm_bits += b; rm_MB += m

            if rm_bits > 0:
                out["frozen_excluded"][key] = {
                    "snapshot_before": snapshot,
                    "removed_bits": rm_bits,
                    "removed_MB": rm_MB,
                    "note": "excluded frozen module(s) from per-content bitrate",
                }
            removed_bits_total += rm_bits
            removed_MB_total   += rm_MB

        _maybe_exclude("den")
        _maybe_exclude("app")

        if removed_bits_total > 0:
            out["total_bits_all"] -= removed_bits_total
            out["total_MB_all"]   -= removed_MB_total
            out["note"] = "One or more codec modules are frozen and excluded from totals."

        return out
