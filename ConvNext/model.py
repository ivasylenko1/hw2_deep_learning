import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import torch


class CustomLayerNorm(nn.Module):
    def __init__(self, normal_shape, eps=1e-6):
        super().__init__()                      
        self.bias = nn.Parameter(torch.zeros(normal_shape))
        self.gamma = nn.Parameter(torch.ones(normal_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False) 
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)        

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
    mask = mask.floor_()  
    return x / keep_prob * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class ConvNextBlock(nn.Module):
    def __init__(self, in_channels,kernel,padding = 3, drop_prob = 0.0): 
         super().__init__()
         self.conv1 = nn.Conv2d(in_channels, in_channels , kernel_size = 7, padding=padding, groups = in_channels)
         self.ln1 = CustomLayerNorm(in_channels)
         self.conv2 = nn.Conv2d(in_channels, 4 * in_channels, 1) 
         self.gelu = nn.GELU()
         self.conv3 = nn.Conv2d(4*in_channels,  in_channels, 1) 
         self.scalar = nn.Parameter(torch.ones(in_channels, 1, 1))
         self.drop_path = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()
    def forward(self, x):
        resid = x
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.gelu(self.conv2(x))
        x = self.scalar * self.conv3(x)
        return resid + self.drop_path(x)

class NeoConvNeXt(nn.Module):
   def __init__(self, out_channels,depths =(3,3,9,3), dims =(96,192,384,768), max_drop = 0.33, input_augment = False):
        super().__init__()
        self.stem = nn.Sequential(nn.Dropout2d(p= 0.15) if input_augment else nn.Identity(),nn.Conv2d(3, dims[0], kernel_size =4, stride = 4), CustomLayerNorm(dims[0]))
        dp_rates = torch.linspace(out_channels, max_drop,  sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        idx = 0
        for i in range(len(depths)):
            blocks = [ConvNextBlock(dims[i], dp_rates[idx + j]) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            idx+=depths[i]
            if i < 3:
                self.downsamples.append(nn.Sequential(
                    CustomLayerNorm(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                ))        
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)  
        self.head = nn.Linear(dims[-1], out_channels)

        self.apply(self._init_weights)
        
   def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.uniform_(m.bias, 0, 0.02)

   def forward(self, x):
        x = self.stem(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsamples[i](x)
        x = x.mean(dim=(2, 3))         
        x = self.head_norm(x)
        return self.head(x)     
        
