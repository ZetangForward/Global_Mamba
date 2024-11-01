'''
Standalone Long Conv class.

The `LongConvModel` class defined in this file provides a simple backbone to train models.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from opt_einsum import contract
from models.modules.mamba_analysis_utils import *
from models.modules.utils import *


class OurModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.1):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class FlashLongconv(OurModule):
    def __init__(
            self,
            config, layer_idx, 
            conv1d_config=None,
            channels=1,
            dropout=0.2,
            kernel_learning_rate=0.001, 
            kernel_lam=0.003, 
            kernel_dropout=0.0,
    ):
        super().__init__()
        self.config = config
        self.conv1d_config = conv1d_config
        self.layer_idx = layer_idx
        self.long_kernel_size  = conv1d_config.get("long_conv_kernel", 128)

        self.H = config.intermediate_size
        self.L = self.long_kernel_size # for causal conv
        self.channels = 1
        self.dropout = nn.Dropout(p=dropout)
        self.kernel_learning_rate = kernel_learning_rate
        self.kernel_lam = kernel_lam
        self.kernel_drop = torch.nn.Dropout(p=kernel_dropout)
        self.use_ma_smoothing = False
        ma_window_len = 5

        self.D = nn.Parameter(torch.randn(channels, self.H))

        # Pointwise
        self.activation = nn.GELU()

        self.kernel = torch.nn.Parameter(torch.randn(self.channels, self.H, self.L) * 0.002) #(c,H,L) 

        dropout_fn = DropoutNd if 1 else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        self.register("kernel", self.kernel, kernel_learning_rate)

        if self.use_ma_smoothing:
            # if smooth_freq:
            #     weight = torch.arange(ma_window_len, dtype = self.kernel.dtype)
            #     weight = torch.exp(-0.5 * torch.abs(weight - ma_window_len // 2) ** 2)
            #     weight = repeat(weight, 'l -> h1 h2 l', h1 = self.H, h2 = 1)
            #     weight = weight.type(torch.fft.rfft(self.kernel).dtype)
            #     self.smooth_weight = weight
            # else:
            self.ma_window_len = ma_window_len
            assert self.ma_window_len%2!=0, "window size must be odd"
            padding = (self.ma_window_len//2)
            self.smooth = torch.nn.AvgPool1d(kernel_size=self.ma_window_len,stride=1,padding=padding)
        
        self.output_linear = LinearActivation(
            self.H * self.channels,
            self.H,
            # self.H*self.channels,
            # self.d_model*(1 if self.gate is None else self.gate),
            transposed=True,
            initializer=None,
            activation='glu',
            activate='gelu',
            weight_norm=False,
        )


    def forward(self, u):
        L = u.size(-1)
        k = self.kernel.to(torch.float32)
        u = u.to(torch.float32)

        # import pdb;pdb.set_trace()

        # smootg operator
        # k = self.smooth(k)
        # squash operator
        k = F.relu(torch.abs(k)-self.kernel_lam)*torch.sign(k)
        k = self.kernel_drop(k)
        
        
        # use FFT to compute convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # Ensure the same length after padding
        u_f = torch.fft.rfft(u, n=2 * L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=2 * L)  # Return to time domain
        y = y[..., :L]

        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')



        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)


        # Transpose for the linear
        # y = y.transpose(-1, -2)
        # y = self.output_linear(y)
        # y = y.transpose(-1, -2)

        return y

