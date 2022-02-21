import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Binarize(nn.Module):
    def __init__(self, th=0.5):
        super().__init__()
        self.th = th
    
    def forward(self, x):
        x = BinarizeFunction.apply(x, self.th)
        return x

class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        output = (input > th).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SearchableConv2d(nn.Module):
    def __init__(self, inplane, outplane, **kwargs):
        super().__init__()
        self.cin = inplane
        self.cout = outplane
        self.alpha = Parameter(torch.Tensor(outplane))
        self.alpha.data.fill_(1.0)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.binarize = Binarize(th=0.5)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups']
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size', torch.tensor(0, dtype=torch.float))
        self.register_buffer('ops', torch.tensor(0, dtype=torch.float))

    def forward(self, x, bin_alpha_in=None):
        in_shape = x.shape
        # Prune weights
        conv = self.conv
        weight = conv.weight
        bin_alpha_out = self.binarize(self.alpha)
        pruned_weight = weight * bin_alpha_out.view(self.cout, 1, 1, 1)
        out = F.conv2d(
            x, pruned_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        # Compute complexities with effective shapes
        if bin_alpha_in is None:
            bin_alpha_in = torch.ones(self.cin)
        #tmp = torch.tensor(
        #    self.param_size * 
        #    (bin_alpha_out.sum() / self.cout) * # Effective Cout
        #    (bin_alpha_in.sum() / self.cin), # Effective Cin
        #    dtype=torch.float)
        self.size = self.param_size * \
            (bin_alpha_out.sum() / self.cout) * \
            (bin_alpha_in.sum() / self.cin)
        self.ops = self.filter_size * \
            (bin_alpha_out.sum() / self.cout) * \
            (bin_alpha_in.sum() / self.cin) * \
            in_shape[-1] * in_shape[-2]
        return out, bin_alpha_out, self.size, self.ops
    
    def complexity_loss(self):
        raise NotImplementedError()

    def fetch_best_arch(self):
        raise NotImplementedError()