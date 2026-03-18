import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt

class Multiscale_waveletconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=3, wt_type='db1', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert in_channels == out_channels

        self.wt_levels = wt_levels
        self.in_channels = in_channels  
        
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)

        self.basic_conv = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=stride, dilation=1, groups=in_channels, bias=bias) for _ in range(wt_levels+1)])
        self.base_scale = nn.ModuleList([_ScaleModule([1,in_channels,1,1]) for _ in range(wt_levels+1)])
        wavelet_convs = []
        wavelet_scales = []
        
        for i in range(wt_levels, 0, -1):
            wavelet_conv = [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(i)]
            wavelet_scale = [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(i)]
            wavelet_convs.append(nn.ModuleList(wavelet_conv))
            wavelet_scales.append(nn.ModuleList(wavelet_scale))

        self.wavelet_convs = nn.ModuleList(wavelet_convs)
        self.wavelet_scales = nn.ModuleList(wavelet_scales)



    def forward(self, x):
        x_chunk = []
        x_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        x_chunk.append([self.base_scale[_](self.basic_conv[_](x)) for _ in  range(self.wt_levels+1)])

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
                
            # curr_x_ll.shape(1, 64, 128, 128) --> curr_x.shape(1, 64, 4, 64, 64) 4:[ll, lh, hl, hh]
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]
            shape_x = curr_x.shape
            # curr_x:(1, 64, 4, 64, 64) reshape--> curr_x_tag:(1, 64*4, 64, 64)
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = [self.wavelet_scales[i][_](self.wavelet_convs[i][_](curr_x_tag)).reshape(shape_x) for _ in range(self.wt_levels-i)]
            x_chunk.append(curr_x_tag)
            # curr_x_tag:(1, 64*4, 64, 64) reshape--> curr_x:(1, 64, 4, 64, 64) 
        return x_chunk, shapes_in_levels
    
class SG_new(nn.Module):
    def __init__(self, in_channels, group=4, groups = 1,SE=False):
        super(SG_new, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.SE=SE
        if SE:
            self.SEs = nn.ModuleList([oneConv(in_channels,in_channels,1,0,1, groups=groups) for _ in range(group)])
        self.softmax = nn.Softmax(dim = 1)
        self.softmax_1 = nn.Sigmoid()

    def forward(self, x): # x:[batch, group, channel, H, W]
        batch_size, group, channels, height, width = x.shape
        
        # Apply global average pooling
        gap_out = self.gap(x.view(-1, channels, height, width))  # Reshape to [batch*group, channels, 1, 1]
        gap_out = gap_out.view(batch_size, group, channels, 1, 1)  # Reshape back to [batch, group, channels, 1, 1]
        
        if self.SE:
            # Apply the SE layers
            weight = [self.SEs[i](gap_out[:, i,:,:,:]) for i in range(group)]
            weight = torch.cat(weight, 2) 

        # weight = self.softmax(self.softmax_1(gap_out))

        gap_out = gap_out.permute(0, 2, 1, 3, 4) # Reshapeto [batch, channels, group, 1, 1]
        gap_out = self.softmax(self.softmax_1(gap_out))
        weight = gap_out.permute(0, 2, 1, 3, 4) # Reshapeto [batch, group, channels, 1, 1]

        x_att = x * weight.view(batch_size, group, channels, 1, 1)  # Broadcasting

        return x_att.sum(dim=1)

class IWT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=1, agent_conv_1=False, agent_conv_2=False, wt_type='db1', *args, **kwargs):
        super(IWT, self).__init__()
        assert in_channels == out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.agent_conv_1 = agent_conv_1
        self.agent_conv_2 = agent_conv_2

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        if agent_conv_2:
            self.agent_convs = nn.ModuleList(
                [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
            )
            self.agent_scale = nn.ModuleList(
                [_ScaleModule([1,in_channels*4,1,1], init_scale=1) for _ in range(self.wt_levels)]
            )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        next_x_ll = 0
        # range(self.wt_levels-1, -1, -1) == [4,3,2,1,0]
        x, shapes_in_levels = x 
        x_ll_in_levels = [x[_][:,:,0,:,:] for _ in range(1,len(x))]
        x_h_in_levels = [x[_][:,:,1:4,:,:] for _ in range(1,len(x))]
        x = x[0]

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            if self.agent_conv_2:
                shape_x_ll = curr_x_ll.shape
                # curr_x:(1, 64, 4, 64, 64) reshape--> curr_x_tag:(1, 64*4, 64, 64)
                curr_x_ll = curr_x_ll.reshape(shape_x_ll[0], shape_x_ll[1], shape_x_ll[3], shape_x_ll[4])
                curr_x_ll = self.agent_ll_conv[i](curr_x_ll)
                # curr_x_tag:(1, 64*4, 64, 64) reshape--> curr_x:(1, 64, 4, 64, 64) 
                curr_x_ll = curr_x_ll.reshape(shape_x_ll)

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        # assert len(x_ll_in_levels) == 0
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
    
class BrokenBlock(nn.Module):

    def __init__(self,  dim_len, dim=1, dim_shape=4, group=1, *args, **kwargs):
        super(BrokenBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.group = group
        self.dim_len = dim_len
        self.view = list(np.ones(dim_shape, int))

    def getShuffle(self, shape):
        # 生成随机索引
        self.view[self.dim] = shape[self.dim]
        perm = torch.randperm(self.dim_len//self.group).unsqueeze(1)
        indices = torch.cat([perm * self.group + _ for _ in range(self.group)], dim=1)
        indices = indices.view(self.view).expand(shape)
        return indices

    def forward(self, x):
        if self.training:
            return torch.gather(x, self.dim, self.getShuffle(x.shape).to(x.device))
        else:
            return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
    
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x