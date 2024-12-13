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
from models.modules.longconv_kernel import *
from models.modules.utils import *



class FlashLongconvLayer(nn.Module):
    def __init__(
            self,
            config, layer_idx, 
            conv1d_config=None,
            channels=1,
            dropout=0.0, tie_dropout=True,
            kernel_learning_rate=0.001, 
            kernel_lam=0.003, 
            kernel_dropout=0.0,
            activation='gelu', # activation between conv and FF
            postact='glu', # activation after FF
            initializer=None,
            weight_norm=False, # weight normalization on FF
            transposed=False, # axis ordering (B, L, D) or (B, D, L)
            **kernel_args,
    ):
        super().__init__()
        self.config = config
        self.conv1d_config = conv1d_config
        self.layer_idx = layer_idx
        self.long_kernel_size  = conv1d_config.get("long_conv_kernel", 128)

        self.H = config.hidden_size
        self.L = self.long_kernel_size # for causal conv
        self.channels = 1
        self.kernel_learning_rate = kernel_learning_rate
        self.kernel_lam = kernel_lam
        self.kernel_drop = torch.nn.Dropout(p=kernel_dropout)
        self.use_ma_smoothing = False
        self.bidirectional = False
        if self.bidirectional:
            channels *= 2

        # import pdb;pdb.set_trace()
        self.D = nn.Parameter(torch.randn(channels, self.H))


        self.kernel = LongConvKernel(self.H, L=self.L, channels=channels, **kernel_args)

        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()


        self.output_linear = LinearActivation(
            self.H * self.channels,
            self.H,
            # self.H*self.channels,
            # self.d_model*(1 if self.gate is None else self.gate),
            transposed=transposed,
            initializer=initializer,
            activation=postact,
            activate=activation,
            weight_norm=weight_norm,
        )

    def forward(self, u):   #  (B, L, H)
        u = u.transpose(-1, -2)
        L = u.size(-1)

        o_dtype = u.dtype
        L_kernel  = L if self.L is None else min(L, self.L )
        k,_ = self.kernel(L=L_kernel)
        k=k.to(torch.float32)
        u = u.to(torch.float32)

         # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))


        k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)
        
        
        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)



        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')


        y = y.transpose(-1, -2)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)
        y = y.to(o_dtype)

        return y

