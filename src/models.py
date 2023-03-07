import torch
import torch.nn as nn
from .model_utils import SpectralNorm

def get_model(config):
    
    if config['img_size'] == 32:
        return ResNet32x32(config)
    elif config['img_size'] == 64:
        if config['model'] == 'resnet':
            return ResNet64x64(config)
        elif config['model'] == 'resnetshallow':
            return ResNet64x64Shallow(config)
    elif config['img_size'] == 128:
        return ResNet128x128(config)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        in_dim = config['in_dim']
        h_dim = config['hidden_dim']
        n_hidden_layers = config['n_hidden_layers']
        sn = config['spectral_norm']

        module_list = []

        module_list.append(SpectralNorm(nn.Linear(in_dim, h_dim), sn))
        module_list.append(nn.LeakyReLU(0.2))

        for _ in range(n_hidden_layers):
            module_list.append(SpectralNorm(nn.Linear(h_dim, h_dim), sn))
            module_list.append(nn.LeakyReLU(0.2))

        module_list.append(SpectralNorm(nn.Linear(h_dim, 1), sn))
        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)


class Self_Attn(nn.Module):
    """
    Self attention from SAGAN. Using the corrected version from
    https://github.com/heykeetae/Self-Attention-GAN/issues/54#issuecomment-842042176
    """
    def __init__(self, inChannels, sn=False, k=8):
        super(Self_Attn, self).__init__()
        embedding_channels = inChannels // k  # C_bar
        self.key      = SpectralNorm(nn.Conv2d(inChannels, embedding_channels, 1), sn)
        self.query    = SpectralNorm(nn.Conv2d(inChannels, embedding_channels, 1), sn)
        self.value    = SpectralNorm(nn.Conv2d(inChannels, embedding_channels, 1), sn)
        self.self_att = SpectralNorm(nn.Conv2d(embedding_channels, inChannels, 1), sn)
        self.gamma    = nn.Parameter(torch.tensor(0.0))
        self.softmax  = nn.Softmax(dim=1)

    def forward(self,x):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """
        batchsize, C, H, W = x.size()
        N = H * W                                       # Number of features
        f_x = self.key(x).view(batchsize,   -1, N)      # Keys                  [B, C_bar, N]
        g_x = self.query(x).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
        h_x = self.value(x).view(batchsize, -1, N)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        v = v.view(batchsize, -1, H, W)                 # Recover input shape   [B, C_bar, H, W]
        o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
        y = self.gamma * o + x                          # Learnable gamma + residual
        return y, o



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sn=False):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.act = nn.LeakyReLU(0.2)
        
        self.conv1 = SpectralNorm(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), sn)
        self.conv2 = SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False), sn)

        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_conv = SpectralNorm(nn.Conv2d(in_planes, self.expansion*planes,
                                kernel_size=1, stride=stride, bias=False), sn)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out += self.shortcut_conv(x)
        out = self.act(out)
        return out


class ResNet32x32(nn.Module):
    def __init__(self, config, block=BasicBlock, num_classes=1):
        super(ResNet32x32, self).__init__()

        self.in_planes = 128
        self.config = config
        self.self_attn = config['self_attn']
        self.act = nn.LeakyReLU(0.2)
        sn = config['spectral_norm']

        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size=3,
                               stride=1, padding=1, bias=False), sn)
        self.layer1 = self._make_layer(block, planes=128, num_blocks=3, stride=1, sn = sn)
        if self.self_attn:
            self.self_attn_module = Self_Attn(128, sn=sn)
        self.layer2 = self._make_layer(block, planes=256, num_blocks=3, stride=2, sn = sn)            
        self.layer3 = self._make_layer(block, planes=256, num_blocks=3, stride=2, sn = sn)
        self.layer4 = self._make_layer(block, planes=256, num_blocks=3, stride=2, sn = sn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = SpectralNorm(nn.Linear(256, num_classes), sn)
    
    def _make_layer(self, block, planes, num_blocks, stride, sn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.layer1(out)
        if self.self_attn:
            out, _ = self.self_attn_module(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.squeeze(-1)

# class ResNet32x32(nn.Module):
#     def __init__(self, config, block=BasicBlock, num_classes=1):
#         super(ResNet32x32, self).__init__()

#         self.in_planes = 128
#         self.config = config
#         self.self_attn = config['self_attn']
#         self.act = nn.LeakyReLU(0.2)
#         sn = config['spectral_norm']

#         self.conv1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size=3,
#                                stride=1, padding=1, bias=False), sn)
#         self.layer1 = self._make_layer(block, planes=128, num_blocks=2, stride=2, sn = sn)
#         if self.self_attn:
#             self.self_attn_module = Self_Attn(128, sn=sn)
#         self.layer2 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)            
#         self.layer3 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = SpectralNorm(nn.Linear(256, num_classes), sn)
    
#     def _make_layer(self, block, planes, num_blocks, stride, sn):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, sn))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.act(out)
#         out = self.layer1(out)
#         if self.self_attn:
#             out, _ = self.self_attn_module(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out.squeeze(-1)


class ResNet64x64(nn.Module):
    def __init__(self, config, block=BasicBlock, num_classes=1):
        super(ResNet64x64, self).__init__()

        self.in_planes = 64
        self.config = config
        self.self_attn = config['self_attn']
        self.act = nn.LeakyReLU(0.2)
        sn = config['spectral_norm']

        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False), sn)
        self.layer1 = self._make_layer(block, planes=64, num_blocks=1, stride=2, sn = sn)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=2, stride=2, sn = sn)
        if self.self_attn:
            self.self_attn_module = Self_Attn(128, sn=sn)     
        self.layer3 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)
        self.layer4 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = SpectralNorm(nn.Linear(256, num_classes), sn)
    
    def _make_layer(self, block, planes, num_blocks, stride, sn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        if self.self_attn:
            out, _ = self.self_attn_module(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.squeeze(-1)


class ResNet128x128(nn.Module):
    def __init__(self, config, block=BasicBlock, num_classes=1):
        super(ResNet128x128, self).__init__()

        self.in_planes = 64
        self.config = config
        self.self_attn = config['self_attn']
        self.act = nn.LeakyReLU(0.2)
        sn = config['spectral_norm']

        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False), sn)
        self.layer1 = self._make_layer(block, planes=64, num_blocks=1, stride=2, sn = sn)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=1, stride=2, sn = sn)
        if self.self_attn:
            self.self_attn_module = Self_Attn(128, sn=sn)     
        self.layer3 = self._make_layer(block, planes=128, num_blocks=2, stride=2, sn = sn)
        self.layer4 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)
        self.layer5 = self._make_layer(block, planes=256, num_blocks=2, stride=2, sn = sn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = SpectralNorm(nn.Linear(256, num_classes), sn)
    
    def _make_layer(self, block, planes, num_blocks, stride, sn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        if self.self_attn:
            out, _ = self.self_attn_module(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.squeeze(-1)

class ResNet64x64Shallow(nn.Module):
    def __init__(self, config, block=BasicBlock, num_classes=1):
        super(ResNet64x64Shallow, self).__init__()

        self.in_planes = 64
        self.config = config
        self.self_attn = config['self_attn']
        self.act = nn.LeakyReLU(0.2)
        sn = config['spectral_norm']

        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False), sn)
        self.layer1 = self._make_layer(block, planes=64, num_blocks=1, stride=2, sn = sn)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=1, stride=2, sn = sn)
        if self.self_attn:
            self.self_attn_module = Self_Attn(128, sn=sn)
        self.layer3 = self._make_layer(block, planes=128, num_blocks=1, stride=2, sn = sn)
        self.layer4 = self._make_layer(block, planes=256, num_blocks=1, stride=2, sn = sn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = SpectralNorm(nn.Linear(256, num_classes), sn)
    
    def _make_layer(self, block, planes, num_blocks, stride, sn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        if self.self_attn:
            out, _ = self.self_attn_module(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.squeeze(-1)    
