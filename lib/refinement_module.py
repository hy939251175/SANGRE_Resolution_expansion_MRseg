import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=15, n_class=7, auxseg=False):
        super(SelFuseFeature, self).__init__()
        
        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    )
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)
        

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.
        
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        grid_ = grid + 0.
        grid[...,0] = 2*grid_[..., 0] / (H-1) - 1
        grid[...,1] = 2*grid_[..., 1] / (W-1) - 1

        # features = []
        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))
        if self.auxseg:
            auxseg = self.auxseg_conv(x)
        else:
            auxseg = None

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return [select_x, auxseg]