import torch
import torch.nn as nn
import torch.nn.functional as F



class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class qfn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        input = torch.sign(input) * (torch.abs(input)**0.5)
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out ** 2
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x
        else:
            weight = torch.tanh(x)
            # weight_q = weight
            # weight_q = qfn.apply(weight, self.wbit)
            weight_q = qfn2.apply(weight, self.wbit)
        return weight_q

class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)

class ConvTranspose2d_Q(nn.ConvTranspose2d):
    def __init__(self, *kargs, **kwargs):
        super(ConvTranspose2d_Q, self).__init__(*kargs, **kwargs)


def conv2d_quantize_fn(bit):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)
            self.bias_q = self.bias

        def forward(self, input, order=None):
            self.weight_q = self.quantize_fn(self.weight)
            # bias_q = self.quantize_fn(self.bias)
            self.bias_q = self.bias
            return F.conv2d(input, self.weight_q, self.bias_q, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_



def transposed_conv2d_quantize_fn(bit):
    class ConvTranspose2d_Q_(ConvTranspose2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=1, groups=1,
                     bias=True, dilation=1):
            super(ConvTranspose2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                                     output_padding, groups, bias, dilation)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)
            self.bias_q = self.bias

        def forward(self, input, order=None):
            self.weight_q = self.quantize_fn(self.weight)
            # bias_q = self.quantize_fn(self.bias)
            self.bias_q = self.bias
            return F.conv_transpose2d(input, self.weight_q, self.bias_q, self.stride, self.padding, self.output_padding,
                                      self.groups, self.dilation)

    return ConvTranspose2d_Q_




def subpel_conv3x3_q(in_ch: int, out_ch: int, r: int = 1, q_bit: int = 32) -> nn.Sequential:
    """quantized 3x3 sub-pixel convolution for up-sampling."""
    Conv2d = conv2d_quantize_fn(q_bit)

    return nn.Sequential(
        Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def transposed_conv3x3_q(in_ch: int, out_ch: int, q_bit: int = 32):
    ConvTranspose2d = transposed_conv2d_quantize_fn(q_bit)

    return ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, output_padding=1, padding=1)


def subpel_conv1x1_q(in_ch: int, out_ch: int, r: int = 1, q_bit: int = 32) -> nn.Sequential:
    """quantized 3x3 sub-pixel convolution for up-sampling."""
    Conv2d = conv2d_quantize_fn(q_bit)

    # Conv2d = conv2d_w4b8_fn(4, 8)

    return nn.Sequential(
        Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )
