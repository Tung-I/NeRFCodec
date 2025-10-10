# lib/plane_codec_dcvc.py  (new file)
import torch
from einops import rearrange
import numpy as np
import torch.nn.functional as F
import math
import cv2
from typing import Tuple, Optional
import subprocess, shutil

DCVC_ALIGN = 32

# ---------------------------------------------------------------------
# FFmpeg via pipes: rawvideo in (BGR24), encoded bytes out; then decode
# ---------------------------------------------------------------------
def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found in PATH. Install ffmpeg or add it to PATH.")

def _ffmpeg_pipe_encode_decode_bgr(
    bgr_f01_hw3: np.ndarray,
    enc_args: list,    # codec-specific args, e.g. ["-c:v","libx265","-x265-params","qp=32",...]
    container: str,    # e.g. "hevc", "ivf", "webm"
) -> tuple[np.ndarray, int]:
    """
    Encode one frame with FFmpeg (pipes, no filesystem), then decode it back.
    Returns: (decoded_bgr_f01, encoded_bits)
    """
    _ensure_ffmpeg()
    assert bgr_f01_hw3.ndim == 3 and bgr_f01_hw3.shape[2] == 3, f"Expected HxWx3, got {bgr_f01_hw3.shape}"
    H, W = int(bgr_f01_hw3.shape[0]), int(bgr_f01_hw3.shape[1])

    # Prepare raw BGR bytes (stdin to encoder)
    bgr_u8 = to_uint8_from_float01(bgr_f01_hw3)
    raw_bytes = bgr_u8.tobytes()

    # ---------------- encode to bytes ----------------
    cmd_enc = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-i", "pipe:0",
        *enc_args,
        "-frames:v", "1",  # single frame
        "-f", container, "pipe:1",
    ]
    enc_proc = subprocess.run(cmd_enc, input=raw_bytes, stdout=subprocess.PIPE, check=True)
    encoded_bytes = enc_proc.stdout
    bits = len(encoded_bytes) * 8

    # ---------------- decode from bytes ----------------
    cmd_dec = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vframes", "1",
        "pipe:1",
    ]
    dec_proc = subprocess.run(cmd_dec, input=encoded_bytes, stdout=subprocess.PIPE, check=True)
    dec_raw = dec_proc.stdout
    # Reinterpret raw bytes into HxWx3 BGR
    if len(dec_raw) != H * W * 3:
        raise RuntimeError(f"Decoded size mismatch: got {len(dec_raw)} bytes, expected {H*W*3}")
    dec_bgr = np.frombuffer(dec_raw, dtype=np.uint8).reshape(H, W, 3)
    dec_f01_bgr = to_float01_from_uint8(dec_bgr)
    return dec_f01_bgr, bits

# -------------------- HEVC (H.265, libx265 QP) --------------------
def hevc_roundtrip_color(
    img_f01_bgr: np.ndarray,
    qp: int = 32,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    enc_args = [
        "-c:v", "libx265",
        "-x265-params", f"qp={int(qp)}",
        "-preset", str(preset),
        "-pix_fmt", str(pix_fmt),
    ]
    return _ffmpeg_pipe_encode_decode_bgr(img_f01_bgr, enc_args, "hevc")

def hevc_roundtrip_mono(
    img_f01: np.ndarray,
    qp: int = 32,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    bgr = np.stack([img_f01, img_f01, img_f01], axis=-1)
    bgr_dec, bits = hevc_roundtrip_color(bgr, qp=qp, preset=preset, pix_fmt=pix_fmt)
    # make it 2D mono again
    return bgr_dec[..., 0], bits

# -------------------- AV1 (libaom-av1 QP) --------------------
# QP range: 0..63 (lower = higher quality)
def av1_roundtrip_color(
    img_f01_bgr: np.ndarray,
    qp: int = 36,
    cpu_used: int = 6,
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    enc_args = [
        "-c:v", "libaom-av1",
        "-b:v", "0",
        "-qp", str(int(qp)),
        "-cpu-used", str(int(cpu_used)),
        "-row-mt", "1",
        "-pix_fmt", str(pix_fmt),
    ]
    return _ffmpeg_pipe_encode_decode_bgr(img_f01_bgr, enc_args, "ivf")

def av1_roundtrip_mono(
    img_f01: np.ndarray,
    qp: int = 36,
    cpu_used: int = 6,
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    bgr = np.stack([img_f01, img_f01, img_f01], axis=-1)
    bgr_dec, bits = av1_roundtrip_color(bgr, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt)
    return bgr_dec[..., 0], bits

# -------------------- VP9 (libvpx-vp9 constant-QP via qmin=qmax) --------------------
# QP range: 0..63 (lower = higher quality)
def vp9_roundtrip_color(
    img_f01_bgr: np.ndarray,
    qp: int = 40,
    cpu_used: int = 4,
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    enc_args = [
        "-c:v", "libvpx-vp9",
        "-b:v", "0",
        "-qmin", str(int(qp)),
        "-qmax", str(int(qp)),
        "-cpu-used", str(int(cpu_used)),
        "-row-mt", "1",
        "-pix_fmt", str(pix_fmt),
    ]
    return _ffmpeg_pipe_encode_decode_bgr(img_f01_bgr, enc_args, "webm")

def vp9_roundtrip_mono(
    img_f01: np.ndarray,
    qp: int = 40,
    cpu_used: int = 4,
    pix_fmt: str = "yuv420p",
) -> tuple[np.ndarray, int]:
    bgr = np.stack([img_f01, img_f01, img_f01], axis=-1)
    bgr_dec, bits = vp9_roundtrip_color(bgr, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt)
    return bgr_dec[..., 0], bits


###############################
def to_uint8_from_float01(img_f01_hw_or_hw3: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255] with rounding; accepts HxW (mono) or HxWx3 (BGR)."""
    img = np.clip(img_f01_hw_or_hw3, 0.0, 1.0)
    return np.rint(img * 255.0).astype(np.uint8)

def to_float01_from_uint8(img_u8_hw_or_hw3: np.ndarray) -> np.ndarray:
    return img_u8_hw_or_hw3.astype(np.float32) / 255.0

def jpeg_roundtrip_color(img_f01_bgr: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """
    JPEG encode/decode round-trip on CPU.
    img_f01_bgr: HxWx3 float in [0,1] (BGR order for OpenCV).
    Returns: (decoded_f01_bgr, encoded_bits)
    """
    img_u8 = to_uint8_from_float01(img_f01_bgr)               # HxWx3 uint8 BGR
    # print(img_u8.shape, img_u8.dtype, img_u8.min(), img_u8.max())
    # raise Exception("stop here")
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg, ...) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)                   # HxWx3 uint8 BGR
    if dec is None:
        raise RuntimeError("cv2.imdecode failed")
    return to_float01_from_uint8(dec), bits

# ======================== CODEC ROUND-TRIPS (mono + color) ========================

def jpeg_roundtrip_mono(img_f01: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """
    JPEG encode/decode for a MONO image.
    img_f01: HxW float in [0,1]
    Returns: (decoded_f01, bits)
    """
    assert img_f01.ndim == 2, f"expected HxW mono, got {img_f01.shape}"
    img_u8 = to_uint8_from_float01(img_f01)             # HxW uint8
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg, mono) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)       # HxW uint8
    if dec is None:
        raise RuntimeError("cv2.imdecode mono failed")
    return to_float01_from_uint8(dec), bits


def png_roundtrip_mono(img_f01: np.ndarray, level: int = 6) -> Tuple[np.ndarray, int]:
    """
    PNG encode/decode for a MONO image (lossless).
    level: 0..9 (higher = smaller, slower). OpenCV default is 3 or 6 depending on build.
    """
    assert img_f01.ndim == 2, f"expected HxW mono, got {img_f01.shape}"
    img_u8 = to_uint8_from_float01(img_f01)             # HxW uint8
    ok, buf = cv2.imencode(".png", img_u8, [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise RuntimeError("cv2.imencode(.png, mono) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)       # HxW uint8
    if dec is None:
        raise RuntimeError("cv2.imdecode mono failed")
    return to_float01_from_uint8(dec), bits


def png_roundtrip_color(img_f01_bgr: np.ndarray, level: int = 6) -> Tuple[np.ndarray, int]:
    """
    PNG encode/decode for COLOR image in BGR float01.
    Provided for completeness; not used when mode='flatten' (we use mono there).
    """
    assert img_f01_bgr.ndim == 3 and img_f01_bgr.shape[2] == 3, f"expected HxWx3, got {img_f01_bgr.shape}"
    img_u8 = to_uint8_from_float01(img_f01_bgr)         # HxWx3 uint8 (BGR)
    ok, buf = cv2.imencode(".png", img_u8, [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise RuntimeError("cv2.imencode(.png, color) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)           # HxWx3 uint8 (BGR)
    if dec is None:
        raise RuntimeError("cv2.imdecode color failed")
    return to_float01_from_uint8(dec), bits


def sandwich_planes_to_rgb(
    x01: torch.Tensor,                            # [T,C,H,W] in [0,1]
    pre_unet: torch.nn.Module,                   # SmallUNet(C->3)
    pre_mlp: torch.nn.Module,                    # MLP 1x1 (C->3)
    bound_pre: torch.nn.Module,                  # BoundedProjector(3)
    align: int,                                  # DCVC_ALIGN
) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    Learned pack: [T,C,H,W] -> y_pad [T,3,Hp,Wp], returns orig (H, W).
    """
    assert x01.dim() == 4, f"expected [T,C,H,W], got {x01.shape}"
    T, C, H, W = x01.shape

    # run per frame; keep it batched over T
    y3 = pre_mlp(x01) + pre_unet(x01)           # [T,3,H,W]
    y01 = bound_pre(y3)                          # [T,3,H,W] in [0,1]

    # pad to multiples of align
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    if pad_h or pad_w:
        y_pad = F.pad(y01, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        y_pad = y01
    return y_pad, (H, W)


def sandwich_rgb_to_planes(
    y_hat: torch.Tensor,                          # [T,3,Hp,Wp], decoder output (float in [0,1])
    orig_size: Tuple[int,int],                    # (H, W) before pad (from sandwich_planes_to_rgb)
    post_unet: torch.nn.Module,                  # SmallUNet(3->C)
    post_mlp: torch.nn.Module,                   # MLP 1x1 (3->C)
    post_bound: Optional[torch.nn.Module] = None # optional BoundedProjector(C)
) -> torch.Tensor:
    """
    Learned unpack: crop pad -> postprocess to [T,C,H,W] (ideally still in [0,1]).
    """
    H, W = orig_size
    y = y_hat[..., :H, :W]                       # [T,3,H,W]

    x_rec = post_mlp(y) + post_unet(y)           # [T,C,H,W]
    if post_bound is not None:
        x_rec = post_bound(x_rec)                # keep it in [0,1] if you prefer
    else:
        # safe clamp for stability since codec can introduce tiny overshoots
        x_rec = x_rec.clamp_(0.0, 1.0)
    return x_rec


# ======================== FEATURE PLANES (C=12) ========================
def pack_planes_to_rgb(
    x: torch.Tensor,
    align: int = DCVC_ALIGN,
    mode: str = "flatten",
    r: int = 4,
    c: int = 4,
):
    """
    x : [T, C, H, W]
    -> y_pad : [T, 3, H2_pad, W2_pad] ; orig : (H2_orig, W2_orig)

    modes:
      - "mosaic":
          * require C % 3 == 0 and C/3 is a perfect square s^2
          * split into 3 groups of Cpc=C/3 channels
          * per group run pixel_shuffle(scale=s) -> [T,1,sH,sW]
          * stack 3 groups as RGB -> [T,3,sH,sW]
      - "flat4":
          * require C % 3 == 0 and C/3 is a perfect square s^2
          * split into 3 groups; per group tile channels as s×s mono -> [T,1,sH,sW]
          * stack 3 groups as RGB -> [T,3,sH,sW]
      - "flatten":
          * tile all channels into an r×c mono canvas -> [T,1,rH,cW]
          * repeat mono to RGB -> [T,3,rH,cW]
    """
    T, C, H, W = x.shape
    if mode not in ("mosaic", "flatten", "flat4"):
        raise ValueError(f"pack: unknown mode '{mode}'")

    if mode in ("mosaic", "flat4"):
        if C % 3 != 0:
            raise ValueError(f"pack({mode}): C must be divisible by 3 (got C={C})")
        Cpc = C // 3
        s = int(math.sqrt(Cpc))
        if s * s != Cpc:
            raise ValueError(
                f"pack({mode}): per-color channels C/3 must be a perfect square; got C/3={Cpc}"
            )
        xg = x.view(T, 3, Cpc, H, W)  # 3 groups

        if mode == "mosaic":
            # per group: [T,Cpc,H,W] --pixel_shuffle(s)--> [T,1,sH,sW]
            tiles = [F.pixel_shuffle(xg[:, g], s) for g in range(3)]
            # keep B,G,R ordering consistent with previous code (use [: : -1] if you want RGB)
            y = torch.cat(tiles[::-1], dim=1)  # [T,3,sH,sW]
            h2, w2 = s * H, s * W
        else:  # flat4 (generalized)
            # tile each group to mono s×s
            mono = [rearrange(xg[:, g], 'T (ry cx) H W -> T 1 (ry H) (cx W)', ry=s, cx=s)
                    for g in range(3)]
            y = torch.cat(mono[::-1], dim=1)  # [T,3,sH,sW]
            h2, w2 = s * H, s * W

    else:  # "flatten"
        if r * c != C:
            raise ValueError(f"pack(flatten): r*c={r*c} must equal C={C}")
        mono = rearrange(x, 'T (ry cx) H W -> T 1 (ry H) (cx W)', ry=r, cx=c)  # [T,1,rH,cW]
        y = mono.repeat(1, 3, 1, 1)                                           # [T,3,rH,cW]
        h2, w2 = r * H, c * W

    # pad to multiples of `align`
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
    return y_pad, (h2, w2)


def unpack_rgb_to_planes(
    y_pad: torch.Tensor,
    C: int,
    orig_size: Tuple[int, int],
    mode: str = "flatten",
    r: int = 4,
    c: int = 4,
):
    """
    Inverse of pack_planes_to_rgb (supports 'mosaic', 'flat4', 'flatten').

    y_pad   : [T,3,H_pad,W_pad]
    C       : target channel count
    orig_size = (H2_orig, W2_orig) as returned by pack (before padding)
    """
    if mode not in ("mosaic", "flatten", "flat4"):
        raise ValueError(f"unpack: unknown mode '{mode}'")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]  # crop the padding

    if mode in ("mosaic", "flat4"):
        if C % 3 != 0:
            raise ValueError(f"unpack({mode}): C must be divisible by 3 (got C={C})")
        Cpc = C // 3
        s = int(math.sqrt(Cpc))
        if s * s != Cpc:
            raise ValueError(
                f"unpack({mode}): per-color channels C/3 must be a perfect square; got C/3={Cpc}"
            )
        # split to B,G,R (pack used tiles[::-1])
        b, g, rch = y.split(1, dim=1)

        if mode == "mosaic":
            # per channel: [T,1,sH,sW] --pixel_unshuffle(s)--> [T,Cpc,H,W] with Cpc=s^2
            blocks = [F.pixel_unshuffle(ch, s) for ch in (rch, g, b)]
            x = torch.cat(blocks, dim=1)  # [T, C, H, W]
            return x
        else:  # flat4 inverse
            # each channel is a mono tiling s×s -> recover s^2 slices
            def _untile(mono_ch):
                # T,1,sH,sW -> T,(s*s),H,W
                return rearrange(mono_ch, 'T 1 (ry H) (cx W) -> T (ry cx) H W', ry=s, cx=s, H=H2//s, W=W2//s)
            blocks = [_untile(ch) for ch in (rch, g, b)]
            x = torch.cat(blocks, dim=1)  # [T, C, H, W]
            return x

    else:  # "flatten"
        if r * c != C:
            raise ValueError(f"unpack(flatten): r*c={r*c} must equal C={C}")
        mono = y[:, :1]
        if H2 % r != 0 or W2 % c != 0:
            raise ValueError(
                f"unpack(flatten): orig_size {(H2,W2)} not divisible by (r,c)=({r},{c})"
            )
        H = H2 // r
        W = W2 // c
        x = rearrange(mono, 'T 1 (ry H) (cx W) -> T (ry cx) H W', ry=r, cx=c, H=H, W=W)
        return x

# def pack_planes_to_rgb(x: torch.Tensor, align: int = DCVC_ALIGN, mode: str = "flatten", r=4, c=4):
#     """
#     x : [T, C, H, W], C == 12
#     -> y_pad : [T, 3, H2_pad, W2_pad] ; orig : (H2_orig, W2_orig)

#     modes:
#       - "mosaic":   3 groups of 4 channels; F.pixel_shuffle(scale=2) per group -> concat as RGB
#       - "flat4":    3 groups of 4 channels; tile each group into 2x2 mono -> concat as RGB
#       - "flatten":  tile channels to a mono 3x4 grid -> repeat to RGB (legacy)
#     """
#     T, C, H, W = x.shape
#     if mode not in ("mosaic", "flatten", "flat4"):
#         raise ValueError(f"pack: unknown mode '{mode}'")

#     if mode == "mosaic":
#         xg = x.view(T, 3, 4, H, W)
#         tiles = [F.pixel_shuffle(xg[:, g], 2) for g in range(3)]  # each [T,1,2H,2W]
#         y = torch.cat(tiles[::-1], dim=1)  # [T,3,2H,2W]  (B,G,R)→RGB-ish
#         h2, w2 = 2 * H, 2 * W

#     elif mode == "flat4":
#         # Tile each 4-ch group as 2x2 mono and map to one RGB channel
#         xg = x.view(T, 3, 4, H, W)                                 # [T,3,4,H,W]
#         mono = [rearrange(xg[:, g], 'T (r c) H W -> T 1 (r H) (c W)', r=2, c=2) for g in range(3)]
#         y = torch.cat(mono[::-1], dim=1)                           # [T,3,2H,2W]
#         h2, w2 = 2 * H, 2 * W

#     else:  # "flatten"
#         mono = rearrange(x, 'T (r c) H W -> T 1 (r H) (c W)', r=r, c=c)  # [T,1,3H,4W]
#         y = mono.repeat(1, 3, 1, 1)                                      # [T,3,3H,4W]
#         h2, w2 = r * H, c * W

#     # pad
#     pad_h = (align - h2 % align) % align
#     pad_w = (align - w2 % align) % align
#     y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
#     return y_pad, (h2, w2)


# def unpack_rgb_to_planes(y_pad: torch.Tensor, C: int, orig_size: Tuple[int, int], mode: str = "flatten", r=4, c=4):
#     """
#     Inverse of pack_planes_to_rgb (kept here unchanged; you’ll extend for flat4 in your next step).
#     """
#     if mode not in ("mosaic", "flatten"):
#         # NOTE: you'll add "flat4" inverse in the follow-up step
#         raise ValueError(f"unpack: unknown mode '{mode}'")

#     H2, W2 = orig_size
#     y = y_pad[..., :H2, :W2]

#     if mode == "mosaic":
#         if C != 12:
#             raise ValueError(f"unpack(mosaic): C must be 12 (got {C})")
#         b, g, r = y.split(1, dim=1)
#         blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]
#         return torch.cat(blocks, dim=1)  # [T,12,H,W]

#     elif mode == "flatten":
#         mono = y[:, :1]
#         if H2 % r != 0 or W2 % c != 0:
#             raise ValueError(f"unpack(flatten): orig_size {(H2,W2)} not divisible by (r,c)=({r},{c})")
#         H = H2 // r; W = W2 // c
#         x = rearrange(mono, 'T 1 (r H) (c W) -> T (r c) H W', r=r, c=c, H=H, W=W)
#         return x

#     else:  # "flatten"
#         raise ValueError(f"unpack: unknown mode '{mode}'")


# ======================== DENSITY (Dz=192) ========================
def pack_density_to_rgb(d5: torch.Tensor, align: int = DCVC_ALIGN, mode: str = "flatten"):
    """
    d5: [1,1,Dy,Dx,Dz] (Dz must be 192 for 'mosaic'/'flat4')
    -> y_pad: [1,3,H2_pad,W2_pad]; orig: (H2_orig,W2_orig)

    modes:
      - "mosaic":
          • Map to [0,1], permute to [1,Dz,Dy,Dx]
          • Split to 3 groups of 64, pixel_shuffle(scale=8) per group -> [1,1,8Dy,8Dx]
          • Concat 3 groups as RGB -> [1,3,8Dy,8Dx]
      - "flat4":
          • Map to [0,1], permute to [1,Dz,Dy,Dx]
          • Split to 3 groups of 64, tile each group into 8x8 mono -> [1,1,8Dy,8Dx]
          • Concat 3 groups as RGB -> [1,3,8Dy,8Dx]
      - "flatten" (legacy, unchanged):
          • Map to [0,1], view as [1,C=Dy,H=Dx,W=Dz], row-wise tile to mono canvas -> repeat to RGB
    """
    assert d5.dim() == 5 and d5.shape[:2] == (1, 1), f"expected [1,1,Dy,Dx,Dz], got {tuple(d5.shape)}"
    _, _, Dy, Dx, Dz = d5.shape

    if mode not in ("flatten", "mosaic", "flat4"):
        raise ValueError(f"pack_density_to_rgb: unknown mode '{mode}'")

    if mode == "flatten":
        d01 = dens_to01(d5)                       # [1,1,Dy,Dx,Dz]
        d01_chw = d01.view(1, Dy, Dx, Dz)         # [1,C=Dy,H=Dx,W=Dz]
        mono, (Hc, Wc) = tile_1xCHW(d01_chw)      # [Hc,Wc]
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,Hc,Wc]
        h2, w2 = Hc, Wc

    else:
        # both "mosaic" and "flat4" need Dz == 192 (3 * 8 * 8)
        if Dz != 192:
            raise ValueError(f"{mode} expects Dz=192, got Dz={Dz}")

        d01 = dens_to01(d5)                                     # [1,1,Dy,Dx,Dz]
        x = d01.permute(0, 1, 4, 2, 3).reshape(1, Dz, Dy, Dx)   # [1,Dz,Dy,Dx]
        xg = x.view(1, 3, 64, Dy, Dx)                           # 3 groups of 64

        if mode == "mosaic":
            # pixel shuffle (scale=8) per group: [1,64,Dy,Dx] -> [1,1,8Dy,8Dx]
            planes = [F.pixel_shuffle(xg[:, g], 8) for g in range(3)]
            y = torch.cat(planes[::-1], dim=1)                  # [1,3,8Dy,8Dx] (B,G,R)→RGB-ish)
        else:  # "flat4"
            # tile 8x8 channels -> [1,1,8Dy,8Dx]
            mono = [rearrange(xg[:, g], 'B (r c) H W -> B 1 (r H) (c W)', r=8, c=8) for g in range(3)]
            y = torch.cat(mono[::-1], dim=1)                    # [1,3,8Dy,8Dx]

        h2, w2 = 8 * Dy, 8 * Dx

    # pad to multiples of `align`
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
    return y_pad, (h2, w2)

def normalize_planes(
        seq, 
        mode="global", 
        global_range=(-20.0, 20.0), 
        eps=1e-6):
    """
    Normalize tri-plane tensor to [0,1].
    seq : [T,C,H,W] (float32/float16)
    Returns:
      seq_n    : normalized to [0,1]
      c_min    : broadcastable min used
      scale    : broadcastable (max-min)
    """
    if mode == "per_channel":
        # per-channel min/max over T,H,W
        c_min = seq.amin(dim=(0, 2, 3), keepdim=True)             # [1,C,1,1]
        c_max = seq.amax(dim=(0, 2, 3), keepdim=True)
    elif mode == "global":
        lo, hi = global_range
        c_min = torch.as_tensor(lo, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
        c_max = torch.as_tensor(hi, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
    else:
        raise ValueError(f"Unknown quant_mode: {mode}")

    scale = (c_max - c_min).clamp_(eps)
    seq_n = ((seq - c_min) / scale).clamp_(0, 1)
    return seq_n, c_min, scale


# ------------------------------------------------------------------

def pad_to_align(x, align=DCVC_ALIGN, mode="replicate"):
    """
    x : [B,3,H,W]  -> pad bottom/right so H,W are multiples of `align`.
    Returns y_pad, (H_orig, W_orig)
    """
    B, C, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    y = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return y, (H, W)


def crop_from_align(x, orig_size):
    """
    x : [B,3,H_pad,W_pad] ; orig_size=(H_orig, W_orig)
    """
    H, W = orig_size
    return x[..., :H, :W]


# ===== Density helpers (packing/unpacking) =====
def dens_to01(d: torch.Tensor) -> torch.Tensor:
    # fixed global mapping used in your packer
    return (d.clamp(-5.0, 30.0) + 5.0) / 35.0

def dens_from01(t01: torch.Tensor) -> torch.Tensor:
    return t01 * 35.0 - 5.0

def tile_1xCHW(feat: torch.Tensor):
    """[1,C,H,W] -> mono canvas [Hc,Wc], row-wise."""
    assert feat.dim() == 4 and feat.size(0) == 1
    _, C, H, W = feat.shape
    tiles_w = int(math.ceil(math.sqrt(C)))
    tiles_h = int(math.ceil(C / tiles_w))
    Hc, Wc = tiles_h * H, tiles_w * W
    canvas = feat.new_zeros(Hc, Wc)
    filled = 0
    for r in range(tiles_h):
        y = 0
        for c in range(tiles_w):
            if filled >= C:
                break
            canvas[r*H:(r+1)*H, y:y+W] = feat[0, filled]
            y += W
            filled += 1
    return canvas, (Hc, Wc)

def untile_to_1xCHW(canvas: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    """inverse of _tile_1xCHW"""
    Hc, Wc = canvas.shape[-2:]
    tiles_h = Hc // H
    tiles_w = Wc // W
    out = canvas.new_zeros(1, C, H, W)
    filled = 0
    for r in range(tiles_h):
        y = 0
        for c in range(tiles_w):
            if filled >= C:
                break
            out[0, filled] = canvas[r*H:(r+1)*H, y:y+W]
            y += W
            filled += 1
    return out