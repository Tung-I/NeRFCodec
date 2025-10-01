import pdb
import warnings

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.zoo.image import model_urls
from compressai.zoo.pretrained import load_pretrained
from compressai.layers import (
    GDN,
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import ScaleHyperprior, MeanScaleHyperprior, Cheng2020Anchor, Cheng2020Attention
from compressai.models.utils import conv, deconv


from .qat import subpel_conv3x3_q, subpel_conv1x1_q, transposed_conv3x3_q

class Adaptor(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, mode: str = "encode"):
        super(Adaptor, self).__init__()
        if mode == "encode":
            self.adaptor = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,)
        else:
            self.adaptor = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2, )

    def forward(self, x):
        return self.adaptor(x)

class AdaptorScaleHyperprior(ScaleHyperprior):
    def __init__(self, plane_ch, feat_ch, latent_ch, q_bit=8, **kwargs):
        super().__init__(feat_ch, latent_ch, **kwargs)
        N, M = feat_ch, latent_ch
        print(f"N, M:{N, M }")

        self.encoder_adaptor = nn.Sequential(
            Adaptor(plane_ch, feat_ch, mode='encode'),
            GDN(N)
        )
        self.decoder_adaptor = nn.Sequential(
            GDN(N, inverse=True),
            subpel_conv3x3_q(N, plane_ch, 2, q_bit=q_bit)
            # Adaptor(feat_ch, plane_ch, mode='decode')
        )

        self.trunc_g_a = nn.Sequential(
            # conv(3, N),
            # GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.trunc_g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            # GDN(N, inverse=True),
            # deconv(N, 3),
        )

    def reload_from_pretrained(self):
        # load analysis transform
        tgt_state_dict = self.trunc_g_a.state_dict()
        src_state_dict = self.g_a[2:].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx - 2

            new_key = old_key.replace(str(module_idx), str(new_module_idx))
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

        # load synthesis transform
        tgt_state_dict = self.trunc_g_s.state_dict()
        src_state_dict = self.g_s[:-2].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx

            new_key = old_key.replace(str(module_idx), str(new_module_idx))
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

    def forward(self, x):
        x_t = self.encoder_adaptor(x)
        y = self.trunc_g_a(x_t)

        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # pdb.set_trace()

        x_t_hat = self.trunc_g_s(y_hat)
        x_hat = self.decoder_adaptor(x_t_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        x_t = self.encoder_adaptor(x)
        y = self.trunc_g_a(x_t)

        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)

        x_t_hat = self.trunc_g_s(y_hat)
        # x_hat = self.decoder_adaptor(x_t_hat).clamp_(0, 1)
        x_hat = self.decoder_adaptor(x_t_hat)

        return {"x_hat": x_hat}

class AdaptorMeanScaleHyperprior(MeanScaleHyperprior):
    def __init__(self, plane_ch, feat_ch, latent_ch, q_bit=8, **kwargs):
        super().__init__(feat_ch, latent_ch, **kwargs)
        N, M = feat_ch, latent_ch
        print(f"N, M:{N, M }")

        self.encoder_adaptor = nn.Sequential(
            Adaptor(plane_ch, feat_ch, mode='encode'),
            GDN(N)
        )
        self.decoder_adaptor = nn.Sequential(
            GDN(N, inverse=True),
            subpel_conv3x3_q(N, plane_ch, 2, q_bit=q_bit)
            # Adaptor(feat_ch, plane_ch, mode='decode')
        )

        self.trunc_g_a = nn.Sequential(
            # conv(3, N),
            # GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.trunc_g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            # GDN(N, inverse=True),
            # deconv(N, 3),
        )

    def reload_from_pretrained(self):
        # load analysis transform
        tgt_state_dict = self.trunc_g_a.state_dict()
        src_state_dict = self.g_a[2:].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx - 2

            new_key = old_key.replace(str(module_idx), str(new_module_idx))
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

        # load synthesis transform
        tgt_state_dict = self.trunc_g_s.state_dict()
        src_state_dict = self.g_s[:-2].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx

            new_key = old_key.replace(str(module_idx), str(new_module_idx))
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

    def forward(self, x):
        x_t = self.encoder_adaptor(x)
        y = self.trunc_g_a(x_t)

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_t_hat = self.trunc_g_s(y_hat)
        x_hat = self.decoder_adaptor(x_t_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        x_t = self.encoder_adaptor(x)
        y = self.trunc_g_a(x_t)

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_t_hat = self.trunc_g_s(y_hat)
        # x_hat = self.decoder_adaptor(x_t_hat).clamp_(0, 1)
        x_hat = self.decoder_adaptor(x_t_hat)
        return {"x_hat": x_hat}

class ChannelChangedScaleHyperprior(ScaleHyperprior):
    def __init__(self, plane_ch, feat_ch, latent_ch, **kwargs):
        super().__init__(feat_ch, latent_ch, **kwargs)
        self.plane_ch = plane_ch
        self.feat_ch = feat_ch
        self.latent_ch = latent_ch

        N, M = feat_ch, latent_ch

        self.g_a = nn.Sequential(
            conv(plane_ch, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, plane_ch),
        )

class AdaptorCheng2020Anchor(Cheng2020Anchor):
    """
    Based on Cheng2020Anchor, add adaptor layers to change input/output channel number
    In Cheng2020Anchor:
        self.g_a, 
        self.h_a, 
        self.h_s, 
        self.g_s
    Added by NeRFCodec:
        self.encoder_adaptor
        self.decoder_adaptor
        self.trunc_g_a
        self.trunc_g_s
    What the receiver actually needs (which are content specific):
        decoder_adaptor
        context_prediction
        entropy_parameters
        entropy_bottleneck
    """
    def __init__(self, plane_ch=16, N=192, q_bit=8, **kwargs):
        super().__init__(N=N, **kwargs)
        # pdb.set_trace()

        print(f"N:{N, N}")

        self.encoder_adaptor = ResidualBlockWithStride(plane_ch, N, stride=2)

        # self.decoder_adaptor = subpel_conv3x3(N, plane_ch, 2) # out_ch(192): 442,368 -> 0.4MB(if INT8)
        #                                                       # out_ch(128): 294,912 -> 0.3MB(if INT8)

        self.decoder_adaptor = subpel_conv3x3_q(N, plane_ch, 2, q_bit=q_bit) # change from 8 -> 6?
        # self.decoder_adaptor = subpel_conv1x1_q(N, plane_ch, 2, q_bit=8)
        # self.decoder_adaptor = transposed_conv3x3_q(N, plane_ch, q_bit=8)
        # self.decoder_adaptor = nn.ConvTranspose2d(N, plane_ch, kernel_size=3, stride=2, output_padding=1, padding=1)

        self.trunc_g_a = nn.Sequential(
            # ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.trunc_g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            # subpel_conv3x3(N, 3, 2),
        )

    def reload_from_pretrained(self):
        # load analysis transform
        tgt_state_dict = self.trunc_g_a.state_dict()
        src_state_dict = self.g_a[1:].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx - 1

            new_key = str(new_module_idx) + old_key[1:]
            renamed_src_state_dict[new_key] = value.clone()

        # pdb.set_trace()
        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

        # load synthesis transform
        tgt_state_dict = self.trunc_g_s.state_dict()
        src_state_dict = self.g_s[:-1].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx

            new_key = str(new_module_idx) + old_key[1:]
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

    def forward(self, x):
        y = self.trunc_g_a(self.encoder_adaptor(x))
        z = self.h_a(y)
        # pdb.set_trace()
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.decoder_adaptor(self.trunc_g_s(y_hat))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.trunc_g_a(self.encoder_adaptor(x))
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.decoder_adaptor(self.trunc_g_s(y_hat))
        return {"x_hat": x_hat}


class AdaptorCheng2020Attention(Cheng2020Attention):
    def __init__(self, plane_ch=16, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        print(f"N:{N, N}")

        self.encoder_adaptor = ResidualBlockWithStride(plane_ch, N, stride=2)

        # self.decoder_adaptor = subpel_conv3x3(N, plane_ch, 2) # out_ch(192): 442,368 -> 0.4MB(if INT8)
        #                                                       # out_ch(128): 294,912 -> 0.3MB(if INT8)

        self.decoder_adaptor = subpel_conv3x3_q(N, plane_ch, 2, q_bit=8)
        # self.decoder_adaptor = nn.ConvTranspose2d(N, plane_ch, kernel_size=3, stride=2, output_padding=1, padding=1)

        self.trunc_g_a = nn.Sequential(
            # ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.trunc_g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            # subpel_conv3x3(N, 3, 2),
        )

    def reload_from_pretrained(self):
        # load analysis transform
        tgt_state_dict = self.trunc_g_a.state_dict()
        src_state_dict = self.g_a[1:].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx - 1

            new_key = str(new_module_idx) + old_key[1:]
            renamed_src_state_dict[new_key] = value.clone()

        # pdb.set_trace()
        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

        # load synthesis transform
        tgt_state_dict = self.trunc_g_s.state_dict()
        src_state_dict = self.g_s[:-1].state_dict()

        renamed_src_state_dict = {}
        for old_key, value in src_state_dict.items():
            module_idx = int(old_key[0])
            new_module_idx = module_idx

            new_key = str(new_module_idx) + old_key[1:]
            renamed_src_state_dict[new_key] = value.clone()

        # missing_keys, unexpected_keys = self.trunc_g_a.load_state_dict(renamed_src_state_dict)
        for (name1, param1), (name2, param2) in zip(renamed_src_state_dict.items(), tgt_state_dict.items()):
            param2.data.copy_(param1.data.clone())

    def forward(self, x):
        y = self.trunc_g_a(self.encoder_adaptor(x))
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.decoder_adaptor(self.trunc_g_s(y_hat))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.trunc_g_a(self.encoder_adaptor(x))
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.decoder_adaptor(self.trunc_g_s(y_hat))
        return {"x_hat": x_hat}