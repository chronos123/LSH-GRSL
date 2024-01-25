import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def ste_round(x):
    return torch.round(x) - x.detach() + x


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


class Conv11(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv11, self).__init__()
        self.model = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

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


class CompressAutoEncoderWinAttnGeluSTE(nn.Module):
    def __init__(self,
                input_nc=3,
                output_nc=3,
                ngf=64,
                n_downsampling=4,
                clamp=True,
                d_range=2,
                finetune=False,
                norm_layer=nn.InstanceNorm2d
                ):

        super().__init__()

        assert d_range == 2, "other models are not provided"

        self.ckp1 = "pretrain_models/pretrained_compnet.pth"

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
    
    def train_res(self, x):
        if self.finetune:
            x = self.model(x)
            x = torch.clamp(x, -self.data_range, self.data_range)
            x = ste_round(x)
            x = x + self.model1(x)
            x = self.upsample(x)
            return x
        else:
            with torch.no_grad():
                x = self.model(x)
                x = torch.clamp(x, -self.data_range, self.data_range)
                x = x.round()
                x = x + self.model1(x)
                x = self.upsample(x)
                return x

    def train_cheng_res(self, x: torch.Tensor):
        if self.finetune:
            x = self.model(x)
            x = torch.clamp(x, -self.data_range, self.data_range)
            x = ste_round(x)
            x = x + self.model1(x)
            x = self.upsample(x)
            return x.detach() - (x * 0.01).detach() + x * 0.01
        else:
            with torch.no_grad():
                x = self.model(x)
                x = torch.clamp(x, -self.data_range, self.data_range)
                x = x.round()
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

    def init_trained_weight(self):
        save_path = self.ckp1
        try:
            self.load_state_dict(torch.load(save_path))
        except:
            state_dict_saved = torch.load(save_path)
            state_dict_net = OrderedDict()
            
            for k, v in state_dict_saved.items():
                state_dict_net[k.replace("module.", "")] = v
            
            self.load_state_dict(state_dict_net)
        print(f"weight from {save_path} loaded")



