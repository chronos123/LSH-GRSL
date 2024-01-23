### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function
import torch.nn.init as init
from collections import OrderedDict
from .mmedit_super_resolution import RDN
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from .balle import ScaleHyperprior

def ste_round(x):
    return torch.round(x) - x.detach() + x


###############################################
# window attention
###########################################

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def window_partition(x, window_size=8):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim=192, window_size=(8, 8), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WinBasedAttention(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=192, num_heads=8, window_size=8, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1)


        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)

        return x


class ResidualUnit(nn.Module):
    """Simple residual unit."""
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(N, N // 2),
            nn.GELU(),
            conv3x3(N // 2, N // 2),
            nn.GELU(),
            conv1x1(N // 2, N),
        )
        self.relu = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out


class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim
        
        self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


# if __name__ == '__main__':
#     x = torch.rand([2, 192, 64, 64])
#     attn = WinBasedAttention()
#     # x = window_partition(x)
#     x = attn(x)
#     print(x.shape)


#################################################
# CBAM module
#####################################################

class Conv11(nn.Module):
    
    def __init__(self, in_c, out_c):
        super(Conv11, self).__init__()
        self.model = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        return self.model(input)
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


##########################################################
# separable conv
##################################################################

class SeperableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1):
        super(SeperableConv2d, self).__init__()
        # depth-wise conv
        self.conv_depth = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False
        )

        # point wise conv
        self.conv_pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=1,
            bias=False
        )
        self.model = nn.Sequential(self.conv_depth, self.conv_pointwise)
        
    def forward(self, input):
        return self.model(input)

##################################################################
# GDN layer
###############################################################

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self,
                 ch,
                 device=torch.device('cuda'),
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, device)
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
    
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "gdn":
        norm_layer = functools.partial(GDN)
        # use inverse=True for encoder; and inverse=False in decoder
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


def define_compG(input_nc,
                 output_nc,
                 ngf,
                 n_downsample_global=3,
                 norm='instance',
                 gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netcompG = CompGenerator(input_nc, output_nc, ngf, n_downsample_global,
                             norm_layer)
    print(netcompG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netcompG.cuda(gpu_ids[0])
    # netcompG.apply(weights_init)
    return netcompG


def define_G(input_nc,
             output_nc,
             ngf,
             netG,
             n_downsample_global=3,
             n_blocks_global=9,
             n_local_enhancers=1,
             n_blocks_local=3,
             norm='instance',
             gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global,
                               n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global,
                             n_blocks_global, n_local_enhancers,
                             n_blocks_local, norm_layer)
    elif netG == "modified":
        netG = GlobalGeneratorModified(input_nc, output_nc, ngf, n_downsample_global,
                                       n_blocks_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    # netG.apply(weights_init)
    return netG


def define_G_GDN(input_nc,
             output_nc,
             ngf,
             netG,
             n_downsample_global=3,
             n_blocks_global=9,
             n_local_enhancers=1,
             n_blocks_local=3,
             norm='gdn',
             gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGeneratorGDN(input_nc, output_nc, ngf, n_downsample_global,
                               n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancerGDN(input_nc, output_nc, ngf, n_downsample_global,
                             n_blocks_global, n_local_enhancers,
                             n_blocks_local, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc,
             ndf,
             n_layers_D,
             norm='instance',
             use_sigmoid=False,
             num_D=1,
             getIntermFeat=False,
             gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer,
                                   use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):

    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()  #for ADE20K (1 to 150 labels + 0 for void)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor,
                                               requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor,
                                               requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                #print(torch.min(pred))
                #print(torch.max(pred))
                #print(target_tensor)
                # assert (pred >= 0.0 & pred <= 1.0).all()
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    ### for texture loss, GRAM MATRIX calculation
    def gram_matrix(self, y):
        #raw_input(y.size())
        #[16, 64, 256, 256])
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        #raw_input(gram)
        return gram

    ###

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i],
                                                     y_vgg[i].detach())

        return loss


class ModifiedVGGLoss(nn.Module):
    
    def __init__(self, gpu_ids=0):
        super(ModifiedVGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights.reverse()

    ### for texture loss, GRAM MATRIX calculation
    @classmethod
    def gram_matrix(cls, y):
        #raw_input(y.size())
        #[16, 64, 256, 256])
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        #raw_input(gram)
        return gram

    ###

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss_texture = []
        for i in range(len(x_vgg)):
            # loss_texture += [self.weights[i] * self.criterion(
            #     self.gram_matrix(x_vgg[i]),
            #     self.gram_matrix(y_vgg[i]).detach()
            # )]
            
            loss += self.weights[i] * self.criterion(x_vgg[i],
                                                     y_vgg[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=32,
                 n_downsample_global=3,
                 n_blocks_global=9,
                 n_local_enhancers=1,
                 n_blocks_local=3,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global,
                                       n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)
                        ]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers - n))
            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                norm_layer(ngf_global),
                nn.ReLU(True),
                nn.Conv2d(ngf_global,
                          ngf_global * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf_global * 2),
                nn.ReLU(True)
            ]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [
                    ResnetBlock(ngf_global * 2,
                                padding_type=padding_type,
                                norm_layer=norm_layer)
                ]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2,
                                   ngf_global,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(ngf_global),
                nn.ReLU(True)
            ]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()
                ]

            setattr(self, 'model' + str(n) + '_1',
                    nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2',
                    nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3,
                                       stride=2,
                                       padding=[1, 1],
                                       count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self,
                                       'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self,
                                     'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers -
                                        n_local_enhancers]
            output_prev = model_upsample(
                model_downsample(input_i) + output_prev)
        return output_prev


class CompGenerator(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=32,
                 n_downsampling=3,
                 norm_layer=nn.BatchNorm2d):
        super(CompGenerator, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * (2**n_downsampling),
                      output_nc,
                      kernel_size=7,
                      padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class GlobalGenerator(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 n_downsampling=3,
                 n_blocks=9,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        #n_downsampling=0
        # input: 1x3xwxh
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), activation
        ]
        # output: 1x64xwxh
        #model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample / NIMA: instead of DS, we feed the downsampled_bic image (1/4)
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2), activation
            ]

        # after 4 downsampling
        # output: 1x128x240x248
        # output: 1x256x120x124
        # output: 1x256x60x62
        # output: 1x1024x30x31

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult,
                            padding_type=padding_type,
                            activation=activation,
                            norm_layer=norm_layer)
            ]

        #n_downsampling=1

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        #model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    # hyper: use_dropout=True, activation=nn.PReLU(), norm_layer=nn.Batchnorm2d
    # 这样可以定义为 Hyper文章中的resblock
    def __init__(self,
                 dim,
                 padding_type,
                 norm_layer,
                 activation=nn.ReLU(True),
                 use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation,
                         use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim), activation
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #print(x.size())
        out = x + self.conv_block(x)
        return out


class MultiscaleDiscriminator(nn.Module):

    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 num_D=3,
                 getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                                       use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3,
                                       stride=2,
                                       padding=[1, 1],
                                       count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self,
                            'scale' + str(num_D - 1 - i) + '_layer' + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[
                nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw),
                nn.Sigmoid()
            ]]
        else:
            sequence += [[
                nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
            ]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class CompGeneratorDeep(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 n_downsampling=4,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super(CompGeneratorDeep, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CompGeneratorGradual(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=32,
                 n_downsampling=3,
                 norm_layer=nn.BatchNorm2d):
        super(CompGeneratorGradual, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      256,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(256,
                      128,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(128,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CompGeneratorDeepAttention(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=32,
                 n_downsampling=3,
                 norm_layer=nn.BatchNorm2d):
        super(CompGeneratorDeepAttention, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
            
        model += [
            CBAM(ngf * (2**n_downsampling))
        ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            CBAM(64)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CompGeneratorDeepOneChannel(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=32,
                 n_downsampling=3,
                 norm_layer=nn.BatchNorm2d):
        super(CompGeneratorDeepOneChannel, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0),
        ]
        
        model += [
            Conv11(3, 1)
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class GlobalGeneratorNoTanh(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 n_downsampling=3,
                 n_blocks=9,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorNoTanh, self).__init__()
        activation = nn.ReLU(True)

        #n_downsampling=0
        # input: 1x3xwxh
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), activation
        ]
        # output: 1x64xwxh
        #model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample / NIMA: instead of DS, we feed the downsampled_bic image (1/4)
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2), activation
            ]

        # after 4 downsampling
        # output: 1x128x240x248
        # output: 1x256x120x124
        # output: 1x256x60x62
        # output: 1x1024x30x31

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult,
                            padding_type=padding_type,
                            activation=activation,
                            norm_layer=norm_layer)
            ]

        #n_downsampling=1

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        #model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CompDeepOneChannelReverse(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc=3,
                 ngf=32,
                 n_downsampling=3,
                 norm_layer=nn.BatchNorm2d):
        super(CompDeepOneChannelReverse, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0)
        ]
        
        model += [
            Conv11(3, 1)
        ]
        
        ## lead to the save of bpp and the complex of the forward
        ## linear reverse cause the compression is linear
        model1 = []
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,
                      64,
                      kernel_size=7,
                      padding=0)
        ]
        
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      3,
                      kernel_size=7,
                      padding=0)
        ]
        
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)

    def forward(self, input):
        one_channel = self.model(input)
        out_color = one_channel + self.model1(one_channel)
        # color restore network
        return out_color
    
    def inference(self, input):
        with torch.no_grad():
            one_channel = self.model(input)
            out_color = one_channel + self.model1(one_channel)
            # color restore network
            return out_color, one_channel


class CompressAutoEncoder(nn.Module):
    def __init__(self,
                input_nc,
                output_nc=3,
                ngf=32,
                n_downsampling=3,
                clamp=False,
                d_range=32767,
                norm_layer=nn.InstanceNorm2d
                ):
        super(CompressAutoEncoder, self).__init__()
        self.output_nc = output_nc
        self.clamp = clamp
        self.data_range = d_range
        activation = nn.ReLU(True)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0)
        ]
        
        model += [
            Conv11(3, 1)
        ]
        
        ## lead to the save of bpp and the complex of the forward
        ## linear reverse cause the compression is linear
        model1 = []
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,
                      64,
                      kernel_size=7,
                      padding=0)
        ]
        
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      3,
                      kernel_size=7,
                      padding=0)
        ]
        
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        
        # upsample x16 ch:512
        self.up4 = Up(16*ngf, 8*ngf, norm_layer, activation)
        # upsample x8 ch:256
        self.up3 = Up(8*ngf, 4*ngf, norm_layer, activation)
        # upsample x4 ch:128
        self.up2 = Up(4*ngf, 2*ngf, norm_layer, activation)
        # upsample x2 ch:64
        self.up1 = Up(2*ngf, ngf, norm_layer, activation)
        
        self.upsample = nn.Sequential(
            Conv11(3, 64),
            norm_layer(64),
            nn.ReLU(True),
            Conv11(64, 512),
            norm_layer(512),
            nn.ReLU(True),
            Conv11(512, 1024),
            norm_layer(1024),
            nn.ReLU(True),
            self.up4,
            self.up3,
            self.up2,
            self.up1,
            Conv11(64, 3)
        )

    def forward(self, input):
        one_channel = self.model(input)
        if self.clamp:
            one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
            # one_channel = one_channel.round()
            # add noise to train
            half = float(0.5)
            noise = torch.empty_like(one_channel).uniform_(-half, half)
            one_channel = one_channel + noise
            
            out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
            return self.upsample(out_color)
            
        out_color = one_channel + self.model1(one_channel)
        # color restore network
        return self.upsample(out_color)
    
    def inference(self, input):
        with torch.no_grad():
            one_channel = self.model(input)
            if self.clamp:
                one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
                one_channel = one_channel.round()
                out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
                return self.upsample(out_color), out_color, one_channel
            
            out_color = one_channel + self.model1(one_channel)
            # color restore network
            return self.upsample(out_color), out_color, one_channel
    
    def train_res(self, input):
        with torch.no_grad():
            x = self.model(x)
            x = x + self.model1(x)
            x = self.upsample(x)
            return x

    @classmethod
    def load_network(cls, network, save_path):
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            state_dict_saved = torch.load(save_path)
            state_dict_net = OrderedDict()
            
            for k, v in state_dict_saved.items():
                state_dict_net[k.replace("module.", "")] = v
            
            network.load_state_dict(state_dict_net)

    def decompress(self, one_channel):
        with torch.no_grad():
            out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
            # color restore network
            return self.upsample(out_color)

    def compress(self, real):
        with torch.no_grad():
            one_channel = self.model(real)
            # 量化
            one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
            one_channel = one_channel.round()
            return one_channel


class CompressAutoEncoder8x(nn.Module):
    def __init__(self,
                input_nc,
                output_nc=3,
                ngf=64,
                n_downsampling=3,
                norm_layer=nn.BatchNorm2d):
        super(CompressAutoEncoder8x, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        model += [
            nn.Conv2d(512,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0)
        ]
        
        model += [
            Conv11(3, 1)
        ]
        
        ## lead to the save of bpp and the complex of the forward
        ## linear reverse cause the compression is linear
        model1 = []
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,
                      64,
                      kernel_size=7,
                      padding=0)
        ]
        
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      3,
                      kernel_size=7,
                      padding=0)
        ]
        
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        
        activation = nn.ReLU(True)

        # upsample x8 ch:256
        self.up3 = Up(8*ngf, 4*ngf, norm_layer, activation)
        # upsample x4 ch:128
        self.up2 = Up(4*ngf, 2*ngf, norm_layer, activation)
        # upsample x2 ch:64
        self.up1 = Up(2*ngf, ngf, norm_layer, activation)
        
        self.upsample = nn.Sequential(
            Conv11(3, 64),
            norm_layer(64),
            nn.ReLU(True),
            Conv11(64, 512),
            norm_layer(512),
            nn.ReLU(True),
            self.up3,
            self.up2,
            self.up1,
            Conv11(64, 3)
        )

    def forward(self, input):
        one_channel = self.model(input)
        out_color = one_channel + self.model1(one_channel)
        # color restore network
        return self.upsample(out_color)
    
    def inference(self, input):
        with torch.no_grad():
            one_channel = self.model(input)
            out_color = one_channel + self.model1(one_channel)
            # color restore network
            return self.upsample(out_color), out_color, one_channel


class Down(nn.Module):
    def __init__(self, input_nc, output_nc, norm, activation,
                 kernal_size=3, stride=2, padding=1):
        super(Down, self).__init__()
        model = []
        model += [
            nn.Conv2d(input_nc,
                output_nc,
                kernel_size=kernal_size,
                stride=stride,
                padding=padding),
            norm(output_nc),
            activation
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class Up(nn.Module):
    def __init__(self, input_nc, output_nc, norm, activation,
                 kernal_size=3, stride=2, padding=1, outpadding=1,
                 padding_type='reflect'
                 ):
        super(Up, self).__init__()
        model = []
        model += [
            nn.ConvTranspose2d(input_nc,
                            output_nc,
                            kernel_size=kernal_size,
                            stride=stride,
                            padding=padding,
                            output_padding=outpadding),
            norm(output_nc),
            activation
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class Mix(nn.Module):
    def __init__(self, input_nc, output_nc, norm, activation,
                 padding_type='reflect'
                 ):
        # concat feature -> attention -> resnet smooth to deblur -> conv1x1 reduce dim
        super(Mix, self).__init__()
        model = []
        model += [CBAM(input_nc)]
        model += [
            ResnetBlock(
                input_nc,
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm
        )
            ]
        model += [Conv11(input_nc, output_nc)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)
    

class CompressAutoEncoderWinAttnGeluSTE(nn.Module):
    def __init__(self,
                input_nc=3,
                output_nc=3,
                ngf=64,
                n_downsampling=4,
                clamp=True,
                d_range=127,
                finetune=False,
                norm_layer=nn.InstanceNorm2d
                ):

        super().__init__()
        self.output_nc = output_nc
        self.clamp = clamp
        self.data_range = d_range
        self.finetune = finetune
        activation = nn.GELU()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation
        ]
        
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activation
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activation
        ]
        model += [
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation
        ]
        model += [
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            norm_layer(1024),
            activation
        ]

        # model += [
        #     Win_noShift_Attention(1024, num_heads=8, window_size=4, shift_size=2)
        #     ]
        
        model += [
            nn.Conv2d(ngf * (2**n_downsampling),
                      256,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            activation,
            Win_noShift_Attention(256, num_heads=8, window_size=4, shift_size=2)
        ]
        
        model += [
            nn.Conv2d(256,
                      64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            activation
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      output_nc,
                      kernel_size=7,
                      padding=0)
        ]
        
        model += [
            Conv11(3, 1)
        ]
        
        ## lead to the save of bpp and the complex of the forward
        ## linear reverse cause the compression is linear
        model1 = []
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1,
                      64,
                      kernel_size=7,
                      padding=0)
        ]
        
        model1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,
                      3,
                      kernel_size=7,
                      padding=0)
        ]
        
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        
        # upsample x16 ch:512
        self.up4 = Up(16*ngf, 8*ngf, norm_layer, activation)
        # upsample x8 ch:256
        self.up3 = Up(8*ngf, 4*ngf, norm_layer, activation)
        # upsample x4 ch:128
        self.up2 = Up(4*ngf, 2*ngf, norm_layer, activation)
        # upsample x2 ch:64
        self.up1 = Up(2*ngf, ngf, norm_layer, activation)
        
        self.upsample = nn.Sequential(
            Conv11(3, 64),
            norm_layer(64),
            activation,
            Conv11(64, 256),
            norm_layer(256),
            activation,
            Win_noShift_Attention(256, num_heads=8, window_size=4, shift_size=2),
            Conv11(256, 1024),
            norm_layer(1024),
            activation,
            # Win_noShift_Attention(1024, num_heads=8, window_size=4, shift_size=2),
            self.up4,
            self.up3,
            self.up2,
            self.up1,
            Conv11(64, 3)
        )

    def forward(self, input):
        one_channel = self.model(input)
        if self.clamp:
            one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
            # one_channel = one_channel.round()
            # add noise to train
            one_channel = ste_round(one_channel)
            out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
            return self.upsample(out_color)
            
        out_color = one_channel + self.model1(one_channel)
        # color restore network
        return self.upsample(out_color)
    
    def inference(self, input):
        with torch.no_grad():
            one_channel = self.model(input)
            if self.clamp:
                one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
                one_channel = one_channel.round()
                out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
                return self.upsample(out_color), out_color, one_channel
            
            out_color = one_channel + self.model1(one_channel)
            # color restore network
            return self.upsample(out_color), out_color, one_channel

    @classmethod
    def load_network(cls, network, save_path):
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            state_dict_saved = torch.load(save_path)
            state_dict_net = OrderedDict()
            
            for k, v in state_dict_saved.items():
                state_dict_net[k.replace("module.", "")] = v
            
            network.load_state_dict(state_dict_net)

    def decompress(self, one_channel):
        with torch.no_grad():
            out_color = one_channel.repeat(1,3,1,1) + self.model1(one_channel)
            # color restore network
            return self.upsample(out_color)

    def compress(self, real):
        with torch.no_grad():
            one_channel = self.model(real)
            # 量化
            one_channel = torch.clamp(one_channel, -self.data_range, self.data_range)
            one_channel = one_channel.round()
            return one_channel    

