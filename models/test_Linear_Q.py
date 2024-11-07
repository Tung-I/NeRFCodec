import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Linear_Q(torch.nn.Linear):
    def __init__(self, w_bit=32, *args, **kwargs):
        super(Linear_Q, self).__init__(*args, **kwargs)
        self.weight_quantize_fn = weight_quantize_fn(w_bit)

    def forward(self, input):
        self.weight_q = self.weight_quantize_fn(self.weight)
        return F.linear(input, self.weight_q, self.bias)

if __name__ == "__main__":
    layer1 = Linear_Q(w_bit=8, in_features=3, out_features=128).to(device)

    input_x = torch.Tensor([1,2,3]).to(device)

    out = layer1(input_x)

    print(layer1.weight[0])
    print(layer1.weight_q[0])

    pdb.set_trace()


